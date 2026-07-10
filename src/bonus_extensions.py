"""Advanced bonus engineering: KTA-optimized (trained) quantum kernels,
device-vs-ideal noise-degradation study, and a blind SMILES prediction
interface. Both quantum-facing sections reuse `quantum_backend.py` /
`quantum_models.py`'s real-execution machinery rather than duplicating any
circuit-running logic -- "noisy" here means an actual `ExecutionConfig(mode=
"aer_noisy")` run (or, with no code changes, `mode="ibm_runtime"` for a
comparison against genuine hardware).
"""

import logging

import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from src.config import RANDOM_SEED
from src.evaluation import kernel_target_alignment
from src.features import compute_descriptors
from src.quantum_backend import ExecutionConfig, QuantumExecutor
from src.quantum_circuits import reuploading_layer
from src.quantum_models import _fidelity_kernel_via_backend

logger = logging.getLogger(__name__)


# --- 1. Kernel Target Alignment optimization ("trained quantum kernel") ----

class KTAOptimizedQuantumKernel:
    """A quantum kernel whose feature map has trainable weight parameters,
    optimized (via COBYLA, real circuit execution throughout) to maximize
    Kernel Target Alignment against the training labels *before* the
    classical SVM head is fit -- a Havlicek-style "trained" quantum kernel
    rather than a fixed encoding.
    """

    def __init__(
        self,
        executor: QuantumExecutor,
        n_layers: int = 1,
        maxiter: int = 40,
        C: float = 1.0,
        random_state: int = RANDOM_SEED,
    ):
        self.executor = executor
        self.n_layers = n_layers
        self.maxiter = maxiter
        self.C = C
        self.random_state = random_state
        self._qc = None
        self._x_params = None
        self._theta_params = None
        self._theta_opt = None
        self._svc = None
        self._X_train = None
        self.kta_before_ = None
        self.kta_after_ = None

    def _kernel_for_theta(self, X: np.ndarray, theta: np.ndarray) -> np.ndarray:
        qc_bound_theta = self._qc.assign_parameters(dict(zip(self._theta_params, theta)))
        return _fidelity_kernel_via_backend(self.executor, qc_bound_theta, self._x_params, X, X, symmetric=True)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KTAOptimizedQuantumKernel":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_features = X.shape[1]
        self._qc, self._x_params, self._theta_params = reuploading_layer(n_features, self.n_layers)
        self._X_train = X

        rng = np.random.RandomState(self.random_state)
        theta0 = rng.uniform(0, 2 * np.pi, size=len(self._theta_params))

        def neg_kta(theta):
            K = self._kernel_for_theta(X, theta)
            return -kernel_target_alignment(K, y)

        self.kta_before_ = -neg_kta(theta0)
        res = minimize(neg_kta, theta0, method="COBYLA", options={"maxiter": self.maxiter, "rhobeg": 0.8})
        self._theta_opt = res.x
        self.kta_after_ = -neg_kta(self._theta_opt)

        K_train = self._kernel_for_theta(X, self._theta_opt)
        self._svc = SVC(kernel="precomputed", C=self.C, probability=True, random_state=self.random_state)
        self._svc.fit(K_train, y)
        logger.info("KTA before optimization: %.4f -> after: %.4f", self.kta_before_, self.kta_after_)
        return self

    def _kernel_to_train(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        qc_bound_theta = self._qc.assign_parameters(dict(zip(self._theta_params, self._theta_opt)))
        return _fidelity_kernel_via_backend(
            self.executor, qc_bound_theta, self._x_params, X, self._X_train, symmetric=False
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._svc.predict(self._kernel_to_train(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._svc.predict_proba(self._kernel_to_train(X))


# --- 2. Noise / real-hardware degradation study ------------------------------

def execution_degradation_study(
    X: np.ndarray,
    y: np.ndarray,
    feature_map_name: str = "angle",
    n_splits: int = 5,
    baseline_config: ExecutionConfig | None = None,
    comparison_config: ExecutionConfig | None = None,
    random_state: int = RANDOM_SEED,
) -> dict:
    """Compare QK-SVM accuracy between a baseline execution target (default:
    ideal local `aer_simulator`) and a comparison target (default: local
    `aer_noisy`, a device-like noise model) under leak-free K-fold CV.

    Passing `comparison_config=ExecutionConfig(mode="ibm_runtime")` runs the
    exact same comparison against genuine IBM Quantum hardware instead of a
    synthetic noise model, with no other code changes -- the whole point of
    routing every kernel evaluation through `quantum_backend.py`.

    K-fold (not the full 29-fold LOOCV used for the core Task 4 suite) is
    used here because a noisy/hardware kernel evaluation costs one real
    circuit execution per pair, per fold -- a deliberate runtime/cost
    tradeoff for this exploratory bonus study, not a shortcut applied to the
    mandatory LOOCV evaluation.
    """
    from src.quantum_models import FEATURE_MAPS

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    baseline_config = baseline_config or ExecutionConfig(mode="aer_simulator")
    comparison_config = comparison_config or ExecutionConfig(mode="aer_noisy")

    baseline_executor = QuantumExecutor(baseline_config)
    comparison_executor = QuantumExecutor(comparison_config)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    baseline_acc, comparison_acc = [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = MinMaxScaler(feature_range=(0.0, np.pi))
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        qc, x_params = FEATURE_MAPS[feature_map_name](X_train_s.shape[1])

        for executor, acc_list in ((baseline_executor, baseline_acc), (comparison_executor, comparison_acc)):
            K_train = _fidelity_kernel_via_backend(executor, qc, x_params, X_train_s, X_train_s, symmetric=True)
            K_test = _fidelity_kernel_via_backend(executor, qc, x_params, X_test_s, X_train_s, symmetric=False)
            svc = SVC(kernel="precomputed", probability=True, random_state=random_state)
            svc.fit(K_train, y_train)
            acc_list.append(svc.score(K_test, y_test))

    return {
        "baseline_label": baseline_config.label(),
        "comparison_label": comparison_config.label(),
        "baseline_accuracy_mean": float(np.mean(baseline_acc)),
        "comparison_accuracy_mean": float(np.mean(comparison_acc)),
        "degradation": float(np.mean(baseline_acc) - np.mean(comparison_acc)),
        "baseline_accuracy_per_fold": baseline_acc,
        "comparison_accuracy_per_fold": comparison_acc,
    }


# --- 3. Blind SMILES prediction interface ------------------------------------

class BlindPredictor:
    """Clean prediction interface for hidden blind evaluation: takes a raw,
    unlabeled SMILES string and returns a Responder/Non-Responder call with
    a confidence score. Wraps an already-fitted model + a scaler fit *only*
    on the 29-drug training set (never refit here, by construction). Works
    with any fitted classical or quantum model exposing `predict_proba`.

    Note: D50 (experimental particle size) is not derivable from SMILES
    structure alone. If the deployed model's feature set includes D50, callers
    must supply it explicitly via `predict(smiles, d50=...)`; omitting it
    raises rather than silently imputing a value.
    """

    def __init__(self, model, scaler: MinMaxScaler, feature_names: list[str]):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names

    def predict(self, smiles: str, d50: float | None = None) -> dict:
        descriptors = compute_descriptors(smiles)
        row = []
        for feat in self.feature_names:
            if feat == "D50":
                if d50 is None:
                    raise ValueError(
                        "D50 is an experimental measurement, not derivable from SMILES; "
                        "pass d50=<value> explicitly."
                    )
                row.append(d50)
            else:
                row.append(descriptors[feat])

        X_row = np.array([row], dtype=float)
        X_scaled = self.scaler.transform(X_row)
        proba = self.model.predict_proba(X_scaled)[0]
        confidence = float(proba[1])
        label = "Responder" if confidence >= 0.5 else "Non-Responder"
        return {"smiles": smiles, "prediction": label, "confidence": confidence}


def fit_deployment_model(model_factory, X, y, feature_range=(0.0, 1.0)) -> tuple:
    """Fit a scaler + model once on the *entire* available dataset (28+1 ->
    all 29), for deployment behind `BlindPredictor`. Distinct from LOOCV,
    which is for evaluation only -- the deployed model should use every
    labeled sample available.
    """
    scaler = MinMaxScaler(feature_range=feature_range)
    X_scaled = scaler.fit_transform(np.asarray(X, dtype=float))
    model = model_factory()
    model.fit(X_scaled, np.asarray(y))
    return model, scaler
