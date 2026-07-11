"""The three bonus features from the challenge brief: the KTA optimized
trained quantum kernel, a noise or real hardware degradation study, and a
blind SMILES prediction interface.

Both quantum facing pieces here reuse the real execution code in
quantum_backend.py and quantum_models.py instead of duplicating any circuit
running logic. "Noisy" means an actual ExecutionConfig with
mode="aer_noisy", and swapping in mode="ibm_runtime" runs the exact same
comparison against real hardware with no other code changes needed.
"""

import logging

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from src.config import RANDOM_SEED
from src.features import compute_descriptors
from src.quantum_backend import ExecutionConfig, QuantumExecutor
from src.quantum_models import TrainedQuantumKernelSVM, _fidelity_kernel_via_backend

logger = logging.getLogger(__name__)

# The KTA optimized trained quantum kernel lives in quantum_models.py now,
# as a regular member of the main LOOCV suite rather than a separate bonus
# only class. It is kept importable under this name here since that is the
# name the "KTA optimization" bonus task refers to, and it avoids having
# the same class defined in two places.
KTAOptimizedQuantumKernel = TrainedQuantumKernelSVM


# --- Noise and real hardware degradation study -------------------------------

def execution_degradation_study(
    X: np.ndarray,
    y: np.ndarray,
    feature_map_name: str = "angle",
    n_splits: int = 5,
    baseline_config: ExecutionConfig | None = None,
    comparison_config: ExecutionConfig | None = None,
    random_state: int = RANDOM_SEED,
) -> dict:
    """Compare QK-SVM accuracy between a baseline execution target (the
    ideal local aer_simulator by default) and a comparison target (a local
    aer_noisy device-like noise model by default), using leak free k-fold
    cross validation.

    Pass comparison_config=ExecutionConfig(mode="ibm_runtime") to run the
    same comparison against real IBM Quantum hardware instead of a
    synthetic noise model. No other changes are needed, since every kernel
    evaluation already goes through quantum_backend.py.

    This uses k-fold cross validation rather than the full 29 fold LOOCV
    used for the main Task 4 suite, because a noisy or real hardware
    kernel evaluation costs one real circuit execution per pair per fold.
    That is a deliberate tradeoff for this exploratory bonus study. It
    does not apply to the required LOOCV evaluation elsewhere.
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


# --- Blind SMILES prediction interface ---------------------------------------

class BlindPredictor:
    """A simple prediction interface for blind evaluation: takes a raw,
    unlabeled SMILES string and returns a Responder or Non-Responder call
    with a confidence score. Wraps an already-fitted model and a scaler
    that was fit only on the 29 drug training set. Neither is ever refit
    here. Works with any fitted classical or quantum model that has a
    predict_proba method.

    D50 (the experimental particle size) cannot be computed from a SMILES
    string alone. If the model's feature list includes D50, callers must
    pass it explicitly with predict(smiles, d50=...). Leaving it out
    raises an error rather than guessing a value.
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
                        "D50 is an experimental measurement and cannot be computed from SMILES. "
                        "Pass d50=<value> explicitly."
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
    """Fit a scaler and a model once on the entire dataset, all 29 drugs,
    for use behind BlindPredictor. This is separate from LOOCV, which is
    only for evaluation. A model that will actually be deployed should be
    trained on every labeled sample available, not held-out folds.
    """
    scaler = MinMaxScaler(feature_range=feature_range)
    X_scaled = scaler.fit_transform(np.asarray(X, dtype=float))
    model = model_factory()
    model.fit(X_scaled, np.asarray(y))
    return model, scaler
