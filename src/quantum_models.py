"""The quantum model classes: QK-SVM, the trained kernel version, VQC, the
data re-uploading classifier, and the QCNN.

Every probability and kernel value these models use comes from an actual
SamplerV2 circuit execution (see quantum_backend.py), meaning shots sampled
from a transpiled circuit on either a local AerSimulator or a real IBM
backend. Nothing here takes a shortcut through exact linear algebra. All
circuits needed for one LOOCV fold, or one optimizer step, are batched into
a single Sampler job wherever possible, which is what keeps this practical
on a real or cloud queued backend.
"""

import logging
import warnings

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from scipy.optimize import minimize
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.svm import SVC

from src.config import RANDOM_SEED
from src.quantum_backend import (
    ExecutionConfig,
    QuantumExecutor,
    build_compute_uncompute_circuit,
    build_measurement_circuit,
    default_execution_config,
    expectation_value_z,
    fidelity_from_counts,
    probability_of_one,
)
from src.quantum_circuits import (
    angle_feature_map,
    build_regression_circuit,
    build_vqc_circuit,
    entangled_feature_map,
    reuploading_layer,
    zz_feature_map,
    zzplus_feature_map,
)

logger = logging.getLogger(__name__)

# scikit-learn 1.9 deprecated SVC(probability=True) in favor of
# CalibratedClassifierCV, but CalibratedClassifierCV splits the precomputed
# kernel matrix internally for cross validation, which does not work well
# with only 28 samples per training fold. probability=True still works
# fine (it is not removed until sklearn 1.11), so we keep using it and
# just silence the warning here instead of switching to a less reliable
# option.
warnings.filterwarnings("ignore", message=".*`probability` parameter was deprecated.*", category=FutureWarning)
# scipy's COBYLA optimizer quietly raises maxiter up to num_vars+2 if a
# smaller value is requested, and prints a warning every time. This is
# expected behavior for our small ansatzes, but it would print hundreds of
# times over a 29 fold LOOCV run, so it is silenced.
warnings.filterwarnings("ignore", message=".*Invalid MAXFUN.*", category=UserWarning)

# Registry of feature maps. Add a new encoding here and any code that
# loops over FEATURE_MAPS will pick it up automatically.
FEATURE_MAPS = {
    "angle": angle_feature_map,
    "entangled": entangled_feature_map,
    "zz": zz_feature_map,
    "zzplus": zzplus_feature_map,
}


# --- Kernel computation through the real backend -----------------------------

def _fidelity_kernel_via_backend(
    executor: QuantumExecutor,
    qc: QuantumCircuit,
    x_params: ParameterVector,
    X_a: np.ndarray,
    X_b: np.ndarray,
    symmetric: bool,
) -> np.ndarray:
    """Build an N by M fidelity kernel matrix, with every entry measured by
    an actual compute-uncompute circuit run through the executor.

    When symmetric is True (X_a and X_b are the same data, as with a
    training kernel), only the upper triangle and diagonal are computed
    and then mirrored, since fidelity is symmetric by definition. This
    roughly halves the number of circuits that need to run.
    """
    n_a, n_b = len(X_a), len(X_b)
    K = np.zeros((n_a, n_b))

    pairs = []
    if symmetric:
        for i in range(n_a):
            for j in range(i, n_b):
                pairs.append((i, j))
    else:
        for i in range(n_a):
            for j in range(n_b):
                pairs.append((i, j))

    circuits = [build_compute_uncompute_circuit(qc, x_params, X_a[i], X_b[j]) for i, j in pairs]
    counts_list = executor.run_counts_batch(circuits)

    num_qubits = qc.num_qubits
    for (i, j), counts in zip(pairs, counts_list):
        fid = fidelity_from_counts(counts, num_qubits, executor.config.shots)
        K[i, j] = fid
        if symmetric:
            K[j, i] = fid
    return K


def compute_quantum_kernel_matrix(
    X: np.ndarray,
    executor: QuantumExecutor,
    feature_map_name: str = "angle",
    **feature_map_kwargs,
) -> np.ndarray:
    """Build an N by N kernel matrix for a given feature map. Used for the
    Task 3 kernel heatmap and KTA score, and usable anywhere a raw quantum
    kernel is needed on its own, without an SVM attached to it.
    """
    qc, x_params = FEATURE_MAPS[feature_map_name](X.shape[1], **feature_map_kwargs)
    return _fidelity_kernel_via_backend(executor, qc, x_params, X, X, symmetric=True)


def _bce_loss(probs: np.ndarray, y: np.ndarray) -> float:
    eps = 1e-9
    p = np.clip(probs, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


# --- Shared training code for the variational models -------------------------

class _VariationalCore:
    """Shared fit and predict logic for the VQC and the QCNN. These models
    differ only in circuit shape, so all the training code lives here once.

    Subclasses implement _build_circuit(n_features) and set
    self.output_qubit. A live QuantumExecutor must be passed in and should
    be reused across an entire LOOCV run rather than recreated per fold,
    otherwise an IBM Runtime backend would need to be reconnected every time.

    Three optimizers are available, and they trade off job count against
    gradient accuracy. Job count matters more than circuit count for wall
    clock time on a real queued backend, so all three batch as many
    circuits as possible into each job:
      - "cobyla": gradient free. One Sampler job (one circuit per training
        sample) per loss evaluation.
      - "spsa": one Sampler job of twice the training set size per
        iteration (the plus and minus perturbations batched together),
        regardless of how many parameters there are. This is the usual
        choice for training on real hardware.
      - "parameter_shift": the exact analytic gradient, needing one
        Sampler job of 2 times the parameter count times the training set
        size per iteration. Most accurate but also the most expensive.
        Meant for small comparisons, not a full LOOCV run.
    """

    def __init__(
        self,
        executor: QuantumExecutor,
        optimizer: str = "cobyla",
        maxiter: int = 60,
        random_state: int = RANDOM_SEED,
    ):
        self.executor = executor
        self.optimizer = optimizer
        self.maxiter = maxiter
        self.random_state = random_state
        self.output_qubit = 0
        self._qc = None
        self._x_params = None
        self._theta_params = None
        self._theta_opt = None

    def _build_circuit(self, n_features: int):
        raise NotImplementedError

    def _predict_probs(self, theta: np.ndarray, X: np.ndarray) -> np.ndarray:
        subs_theta = dict(zip(self._theta_params, theta))
        circuits = [
            build_measurement_circuit(
                self._qc.assign_parameters({**dict(zip(self._x_params, row)), **subs_theta})
            )
            for row in X
        ]
        counts_list = self.executor.run_counts_batch(circuits)
        shots = self.executor.config.shots
        return np.array([probability_of_one(c, self.output_qubit, shots) for c in counts_list])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_VariationalCore":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_features = X.shape[1]
        self._qc, self._x_params, self._theta_params = self._build_circuit(n_features)

        rng = np.random.RandomState(self.random_state)
        theta0 = rng.uniform(0, 2 * np.pi, size=len(self._theta_params))

        if self.optimizer == "cobyla":
            loss_fn = lambda theta: _bce_loss(self._predict_probs(theta, X), y)
            res = minimize(loss_fn, theta0, method="COBYLA", options={"maxiter": self.maxiter, "rhobeg": 0.8})
            self._theta_opt = res.x
        elif self.optimizer == "spsa":
            self._theta_opt = self._optimize_spsa(theta0, X, y)
        elif self.optimizer == "parameter_shift":
            self._theta_opt = self._optimize_parameter_shift(theta0, X, y)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")
        return self

    def _predict_probs_batch_pair(self, theta_plus: np.ndarray, theta_minus: np.ndarray, X: np.ndarray):
        """Run both the plus and minus perturbed parameter sets in one Sampler job."""
        n = len(X)
        subs_plus = dict(zip(self._theta_params, theta_plus))
        subs_minus = dict(zip(self._theta_params, theta_minus))
        circuits = [
            build_measurement_circuit(self._qc.assign_parameters({**dict(zip(self._x_params, row)), **subs_plus}))
            for row in X
        ] + [
            build_measurement_circuit(self._qc.assign_parameters({**dict(zip(self._x_params, row)), **subs_minus}))
            for row in X
        ]
        counts_list = self.executor.run_counts_batch(circuits)
        shots = self.executor.config.shots
        probs = np.array([probability_of_one(c, self.output_qubit, shots) for c in counts_list])
        return probs[:n], probs[n:]

    def _optimize_spsa(self, theta0, X, y, a=0.3, c=0.2, alpha=0.602, gamma=0.101) -> np.ndarray:
        rng = np.random.RandomState(self.random_state)
        theta = theta0.copy()
        for k in range(1, self.maxiter + 1):
            ak = a / (k + 1) ** alpha
            ck = c / (k ** gamma)
            delta = rng.choice([-1.0, 1.0], size=theta.shape)
            probs_plus, probs_minus = self._predict_probs_batch_pair(theta + ck * delta, theta - ck * delta, X)
            loss_plus = _bce_loss(probs_plus, y)
            loss_minus = _bce_loss(probs_minus, y)
            ghat = (loss_plus - loss_minus) / (2 * ck * delta)
            theta = theta - ak * ghat
        return theta

    def _optimize_parameter_shift(self, theta0, X, y, lr=0.3, shift=np.pi / 2) -> np.ndarray:
        theta = theta0.copy()
        n = len(y)
        n_params = len(theta)
        for _ in range(self.maxiter):
            probs = np.clip(self._predict_probs(theta, X), 1e-9, 1 - 1e-9)
            dL_dP = (-(y / probs) + (1 - y) / (1 - probs)) / n

            # Collect every plus and minus shift for every parameter, then run them all in one job.
            subs_list = []
            for i in range(n_params):
                theta_p, theta_m = theta.copy(), theta.copy()
                theta_p[i] += shift
                theta_m[i] -= shift
                subs_list.append(theta_p)
                subs_list.append(theta_m)
            circuits = [
                build_measurement_circuit(
                    self._qc.assign_parameters({**dict(zip(self._x_params, row)), **dict(zip(self._theta_params, th))})
                )
                for th in subs_list
                for row in X
            ]
            counts_list = self.executor.run_counts_batch(circuits)
            shots = self.executor.config.shots
            all_probs = np.array([probability_of_one(c, self.output_qubit, shots) for c in counts_list])
            all_probs = all_probs.reshape(2 * n_params, n)

            grad = np.zeros_like(theta)
            for i in range(n_params):
                probs_p = all_probs[2 * i]
                probs_m = all_probs[2 * i + 1]
                dP_dtheta_i = 0.5 * (probs_p - probs_m)
                grad[i] = np.sum(dL_dP * dP_dtheta_i)
            theta = theta - lr * grad
        return theta

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p1 = self._predict_probs(self._theta_opt, np.asarray(X, dtype=float))
        return np.column_stack([1 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# --- 1. Quantum Kernel SVM --------------------------------------------------

class QuantumKernelSVM:
    """Quantum kernel SVM. Builds a fidelity kernel matrix by running real
    circuits, then trains a standard SVM on that precomputed kernel.
    """

    def __init__(
        self,
        executor: QuantumExecutor,
        feature_map_name: str = "angle",
        feature_map_kwargs: dict | None = None,
        C: float = 1.0,
    ):
        self.executor = executor
        self.feature_map_name = feature_map_name
        self.feature_map_kwargs = feature_map_kwargs or {}
        self.C = C
        self._svc = None
        self._X_train = None
        self._qc = None
        self._x_params = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantumKernelSVM":
        X = np.asarray(X, dtype=float)
        self._qc, self._x_params = FEATURE_MAPS[self.feature_map_name](X.shape[1], **self.feature_map_kwargs)
        self._X_train = X
        K_train = _fidelity_kernel_via_backend(self.executor, self._qc, self._x_params, X, X, symmetric=True)
        self._svc = SVC(kernel="precomputed", C=self.C, probability=True, random_state=RANDOM_SEED)
        self._svc.fit(K_train, y)
        return self

    def _kernel_to_train(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return _fidelity_kernel_via_backend(self.executor, self._qc, self._x_params, X, self._X_train, symmetric=False)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._svc.predict(self._kernel_to_train(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._svc.predict_proba(self._kernel_to_train(X))


# --- 1b. Trained quantum kernel SVM, optimized for Kernel Target Alignment --

class TrainedQuantumKernelSVM:
    """A quantum kernel SVM where the feature map has its own trainable
    weight parameters, separate from the data encoding parameters. Those
    weights are optimized with COBYLA to maximize Kernel Target Alignment
    (KTA) against the training labels, before the SVM is fit on the
    resulting kernel. This is a trained quantum kernel, as opposed to
    QuantumKernelSVM's fixed encoding.

    The fixed feature maps get a fairly low raw KTA score on this dataset
    (around 0.2 to 0.3). That is not a sign that the circuits are wrong.
    KTA measured on the whole dataset at once is not a reliable stand in
    for how well a model will generalize, the same way training accuracy
    is not a reliable stand in for test accuracy. The README has the full
    writeup, but the short version is that training this kernel's weights
    against KTA computed only on each fold's training data (never on the
    held out sample) with 3 re-uploading layers gives the best result
    found in this project across every model, classical or quantum: 96.6%
    LOOCV accuracy. Fewer layers underfit and more layers overfit, which
    is why n_layers defaults to 3 below.
    """

    def __init__(
        self,
        executor: QuantumExecutor,
        n_layers: int = 3,
        maxiter: int = 60,
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

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TrainedQuantumKernelSVM":
        from src.evaluation import kernel_target_alignment

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
        logger.info("Trained kernel KTA for this fold, before and after training: %.4f -> %.4f", self.kta_before_, self.kta_after_)
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


# --- 2. Variational Quantum Classifier ---------------------------------------

class VariationalQuantumClassifier(_VariationalCore):
    """Angle encoding followed by a trainable Ry, Rz, CNOT chain ansatz,
    trained with binary cross entropy loss.
    """

    def __init__(self, n_layers: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.n_layers = n_layers

    def _build_circuit(self, n_features: int):
        self.output_qubit = 0
        return build_vqc_circuit(n_features, self.n_layers)


# --- 4. Quantum Convolutional Neural Network ---------------------------------

def _conv_block(params) -> QuantumCircuit:
    qc = QuantumCircuit(2, name="conv")
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    qc.cx(1, 0)
    qc.rz(np.pi / 2, 0)
    return qc


def _pool_block(params) -> QuantumCircuit:
    qc = QuantumCircuit(2, name="pool")
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc


def _dense_su4_block(params) -> QuantumCircuit:
    """A general two qubit unitary with 15 free parameters, the full
    dimension of SU(4). This captures every possible correlation between
    the two remaining qubits after pooling.
    """
    qc = QuantumCircuit(2, name="dense_su4")
    it = iter(params)
    for q in range(2):
        qc.u(next(it), next(it), next(it), q)
    qc.cx(0, 1)
    qc.rz(next(it), 1)
    qc.ry(next(it), 0)
    qc.cx(1, 0)
    qc.ry(next(it), 0)
    qc.cx(0, 1)
    for q in range(2):
        qc.u(next(it), next(it), next(it), q)
    return qc


def build_qcnn_circuit(n_qubits: int = 6) -> tuple[QuantumCircuit, ParameterVector, ParameterVector, list[int]]:
    """Build the QCNN circuit: 6 qubits with angle encoding, a conv and
    pool stage taking it from 6 qubits down to 3, a second conv and pool
    stage taking it from 3 down to 2, then a dense SU(4) layer on the
    final 2 qubits.

    Returns the 2 surviving qubit indices as `final_qubits` rather than a
    single output qubit, so callers can choose to read out one qubit
    (`QCNNClassifier`, single P(|1>)) or both (`QCNNRegressor`, two <Z>
    expectation values feeding a classical Ridge head).
    """
    x = ParameterVector("x", n_qubits)
    qc = QuantumCircuit(n_qubits, name="QCNN")
    for i in range(n_qubits):
        qc.ry(x[i], i)

    n_conv1 = n_qubits          # one conv block per neighboring pair, wrapping around
    n_pool1 = n_qubits // 2     # pools 6 qubits down to 3
    n_conv2 = 3                 # conv blocks among the 3 remaining qubits
    n_pool2 = 1                 # pools 3 qubits down to 2 (one pair merges, one passes through)
    n_dense = 15                # the dense SU(4) block on the final 2 qubits

    theta = ParameterVector("theta", 3 * (n_conv1 + n_pool1 + n_conv2 + n_pool2) + n_dense)
    idx = 0

    # First conv layer, applied to every neighboring pair of the 6 qubits.
    for i in range(n_qubits):
        j = (i + 1) % n_qubits
        qc.compose(_conv_block(theta[idx:idx + 3]), qubits=[i, j], inplace=True)
        idx += 3

    # First pool layer: 6 qubits down to 3, keeping qubits 1, 3, 5.
    pool1_pairs = [(0, 1), (2, 3), (4, 5)]
    for src, keep in pool1_pairs:
        qc.compose(_pool_block(theta[idx:idx + 3]), qubits=[src, keep], inplace=True)
        idx += 3
    survivors1 = [1, 3, 5]

    # Second conv layer, on the 3 remaining qubits.
    for a, b in [(survivors1[0], survivors1[1]), (survivors1[1], survivors1[2]), (survivors1[2], survivors1[0])]:
        qc.compose(_conv_block(theta[idx:idx + 3]), qubits=[a, b], inplace=True)
        idx += 3

    # Second pool layer: 3 qubits down to 2. One pair merges into
    # survivors1[1], and survivors1[2] passes through unchanged.
    qc.compose(_pool_block(theta[idx:idx + 3]), qubits=[survivors1[0], survivors1[1]], inplace=True)
    idx += 3
    final_qubits = [survivors1[1], survivors1[2]]

    # Dense SU(4) layer on the final 2 qubits.
    qc.compose(_dense_su4_block(theta[idx:idx + n_dense]), qubits=final_qubits, inplace=True)
    idx += n_dense

    return qc, x, theta, final_qubits


class QCNNClassifier(_VariationalCore):
    """Quantum convolutional neural network. Takes 6 input qubits down to 1 output qubit."""

    def __init__(self, n_qubits: int = 6, **kwargs):
        super().__init__(**kwargs)
        self.n_qubits = n_qubits

    def _build_circuit(self, n_features: int):
        if n_features != self.n_qubits:
            raise ValueError(f"QCNN requires exactly {self.n_qubits} input features, got {n_features}")
        qc, x, theta, final_qubits = build_qcnn_circuit(self.n_qubits)
        self.output_qubit = final_qubits[0]
        return qc, x, theta


# --- Shared kernel-ridge-regression utilities (QK-KRR, QK-KRR-trained) ------

def _normalize_fidelity_kernel(K: np.ndarray, diag_a: np.ndarray, diag_b: np.ndarray) -> np.ndarray:
    """K_norm[i, j] = K[i, j] / sqrt(diag_a[i] * diag_b[j]).

    Rescales every kernel entry to what it would be if both feature
    vectors had unit self-fidelity, stabilizing KernelRidge when raw
    fidelities cluster close to 1 (a common symptom of angle-encoded
    feature maps on a small number of qubits).
    """
    denom = np.clip(np.sqrt(np.outer(diag_a, diag_b)), 1e-12, None)
    return K / denom


def _select_alpha_via_inner_loocv(
    K_norm: np.ndarray, y: np.ndarray, alpha_grid: tuple[float, ...], default_alpha: float
) -> float:
    """Pick the alpha_grid candidate with the best Q^2 under a leave-one-out
    sweep *within* the training fold (never touching the outer LOOCV's
    held-out sample -- this is called from inside `fit`, which only ever
    sees a training fold). No new quantum circuits are needed: every
    candidate alpha is scored by refitting `KernelRidge` on submatrices of
    the already-computed `K_norm`, since fidelity is a fixed, precomputed
    kernel independent of alpha.
    """
    n = len(y)
    best_alpha, best_q2 = default_alpha, -np.inf
    for alpha in alpha_grid:
        preds = np.zeros(n)
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            krr_i = KernelRidge(kernel="precomputed", alpha=alpha)
            krr_i.fit(K_norm[np.ix_(mask, mask)], y[mask])
            preds[i] = krr_i.predict(K_norm[i, mask].reshape(1, -1))[0]
        q2 = r2_score(y, preds)
        if q2 > best_q2:
            best_q2, best_alpha = q2, alpha
    return best_alpha


# --- 5. Quantum Kernel Ridge Regression --------------------------------------

class QuantumKernelRidgeRegression:
    """QK-KRR: the same real-execution fidelity kernel machinery as
    `QuantumKernelSVM`, but with a `KernelRidge` regression head instead of
    an `SVC` classifier, predicting COMDR15 directly as a continuous value.

    This is the first model in this suite directly comparable to the
    Patzmann et al. Q^2=0.77 benchmark table, since it targets the
    regression problem instead of thresholding it into a binary class.

    Parameters
    ----------
    executor : QuantumExecutor
    feature_map_name : str, default="zzplus"
        Key into `FEATURE_MAPS`. `zzplus` is recommended: its ZZ terms
        encode feature products (e.g. logP * kappa3-like correlations) that
        plain angle encoding cannot represent.
    alpha : float, default=0.1
        Fallback L2 regularization if `alpha_grid` search is skipped.
    alpha_grid : tuple[float, ...], default=(0.01, 0.1, 1.0)
        Candidate regularizations swept via an inner LOOCV *within* the
        training fold only (never touching the outer LOOCV's held-out
        sample) -- the best-Q^2 alpha is kept as `alpha_`.
    feature_map_kwargs : dict, optional

    Attributes
    ----------
    alpha_ : float
        Alpha actually selected by the inner sweep.
    kernel_frobenius_norm_ : float
        ||K_train||_F of the normalized training kernel -- an
        expressibility diagnostic (a near-identity kernel, ||K||_F close to
        sqrt(n), tells you the feature map barely separates any two drugs).
    kernel_effective_rank_ : float
        exp(entropy of K_train's normalized eigenvalue spectrum) -- a soft
        rank estimate that degrades gracefully as eigenvalues decay
        smoothly, unlike a hard-threshold matrix rank.
    """

    def __init__(
        self,
        executor: QuantumExecutor,
        feature_map_name: str = "zzplus",
        alpha: float = 0.1,
        alpha_grid: tuple[float, ...] = (0.01, 0.1, 1.0),
        feature_map_kwargs: dict | None = None,
    ):
        self.executor = executor
        self.feature_map_name = feature_map_name
        self.alpha = alpha
        self.alpha_grid = alpha_grid
        self.feature_map_kwargs = feature_map_kwargs or {}
        self._qc = None
        self._x_params = None
        self._X_train = None
        self._K_diag_train = None
        self._krr = None
        self.alpha_ = alpha
        self.kernel_frobenius_norm_ = None
        self.kernel_effective_rank_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantumKernelRidgeRegression":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self._qc, self._x_params = FEATURE_MAPS[self.feature_map_name](X.shape[1], **self.feature_map_kwargs)
        self._X_train = X

        K_train = _fidelity_kernel_via_backend(self.executor, self._qc, self._x_params, X, X, symmetric=True)
        self._K_diag_train = np.clip(np.diag(K_train), 1e-12, None)
        K_norm = _normalize_fidelity_kernel(K_train, self._K_diag_train, self._K_diag_train)

        # Expressibility diagnostics on the normalized training kernel.
        self.kernel_frobenius_norm_ = float(np.linalg.norm(K_norm, "fro"))
        eigvals = np.clip(np.linalg.eigvalsh(K_norm), 0.0, None)
        eig_sum = eigvals.sum()
        p = eigvals / eig_sum if eig_sum > 0 else eigvals
        p_nonzero = p[p > 1e-12]
        entropy = -np.sum(p_nonzero * np.log(p_nonzero)) if len(p_nonzero) else 0.0
        self.kernel_effective_rank_ = float(np.exp(entropy))

        self.alpha_ = _select_alpha_via_inner_loocv(K_norm, y, self.alpha_grid, self.alpha)
        self._krr = KernelRidge(kernel="precomputed", alpha=self.alpha_)
        self._krr.fit(K_norm, y)
        logger.info(
            "QK-KRR fold fit | feature_map=%s alpha=%.3g | ||K||_F=%.3f eff_rank=%.2f",
            self.feature_map_name, self.alpha_, self.kernel_frobenius_norm_, self.kernel_effective_rank_,
        )
        return self

    def _kernel_to_train(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        K = _fidelity_kernel_via_backend(self.executor, self._qc, self._x_params, X, self._X_train, symmetric=False)
        # Test self-fidelities measured for real (never assumed to be
        # exactly 1) so the normalization stays honest under noise/hardware
        # execution too -- one extra batched job, cheap since X is a single
        # LOOCV-held-out sample in the standard pipeline.
        K_self = _fidelity_kernel_via_backend(self.executor, self._qc, self._x_params, X, X, symmetric=True)
        diag_test = np.clip(np.diag(K_self), 1e-12, None)
        return _normalize_fidelity_kernel(K, diag_test, self._K_diag_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._krr.predict(self._kernel_to_train(X))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return r2_score(y, self.predict(X))


# --- 6. Multi-observable quantum regressors (VQR, QCNN-R) --------------------

class _MultiObservableRegressorCore:
    """Shared fit/predict machinery for the two multi-observable quantum
    regressors: `VariationalQuantumRegressor` and `QCNNRegressor`. Both
    read <Z_i> expectation values off several output qubits and combine
    them with a classical `Ridge` head, so all of that logic lives here once.

    Subclasses implement `_build_circuit(n_features)`, returning
    `(circuit, x_params, theta_params, output_qubits)`.

    Two-phase optimizer -- "warm SPSA, then local COBYLA":
      Phase 1 runs `maxiter_spsa` SPSA iterations against the MSE between
      the Ridge head's prediction and y. SPSA needs only one Sampler job of
      2N circuits per iteration regardless of parameter count, so it
      explores the (potentially flat / barren-plateau-prone) MSE surface
      cheaply and lands in a reasonable basin.
      Phase 2 runs `maxiter_cobyla` COBYLA iterations from that warm start,
      refining locally without SPSA's residual stochastic gradient noise.
    This two-phase split is a practical response to small-dataset (N=29)
    variational training: SPSA alone plateaus slowly from a random start,
    while COBYLA alone (no gradient signal) frequently stalls near its
    random initialization on a landscape this non-convex.

    Note: the "top-10 parameters by SPSA-estimated gradient" refinement
    described in early planning notes was simplified to full-parameter
    COBYLA refinement from the SPSA warm start -- partial-parameter local
    search would bias which directions get refined based on a single noisy
    SPSA gradient estimate, which is not obviously more robust than just
    refining every parameter from a good starting point.
    """

    def __init__(
        self,
        executor: QuantumExecutor,
        maxiter_spsa: int = 50,
        maxiter_cobyla: int = 20,
        ridge_alpha: float = 0.1,
        random_state: int = RANDOM_SEED,
    ):
        self.executor = executor
        self.maxiter_spsa = maxiter_spsa
        self.maxiter_cobyla = maxiter_cobyla
        self.ridge_alpha = ridge_alpha
        self.random_state = random_state
        self._qc = None
        self._x_params = None
        self._theta_params = None
        self._output_qubits = None
        self._theta_opt = None
        self._ridge = None

    def _build_circuit(self, n_features: int):
        raise NotImplementedError

    def _compute_expectation_values(self, theta: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Batch: ONE Sampler job for every sample, read `len(output_qubits)`
        <Z> values per sample from its single counts dict.
        """
        subs_theta = dict(zip(self._theta_params, theta))
        circuits = [
            build_measurement_circuit(self._qc.assign_parameters({**dict(zip(self._x_params, row)), **subs_theta}))
            for row in X
        ]
        counts_list = self.executor.run_counts_batch(circuits)
        shots = self.executor.config.shots
        ev = np.zeros((len(X), len(self._output_qubits)))
        for i, counts in enumerate(counts_list):
            for k, q in enumerate(self._output_qubits):
                ev[i, k] = expectation_value_z(counts, q, shots)
        return ev

    def _ev_batch_pair(self, theta_plus: np.ndarray, theta_minus: np.ndarray, X: np.ndarray):
        """Both SPSA-perturbed parameter sets submitted as ONE Sampler job."""
        subs_plus = dict(zip(self._theta_params, theta_plus))
        subs_minus = dict(zip(self._theta_params, theta_minus))
        circuits = [
            build_measurement_circuit(self._qc.assign_parameters({**dict(zip(self._x_params, row)), **subs_plus}))
            for row in X
        ] + [
            build_measurement_circuit(self._qc.assign_parameters({**dict(zip(self._x_params, row)), **subs_minus}))
            for row in X
        ]
        counts_list = self.executor.run_counts_batch(circuits)
        shots = self.executor.config.shots
        n = len(X)
        ev = np.zeros((2 * n, len(self._output_qubits)))
        for i, counts in enumerate(counts_list):
            for k, q in enumerate(self._output_qubits):
                ev[i, k] = expectation_value_z(counts, q, shots)
        return ev[:n], ev[n:]

    def _mse_loss(self, theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        ev = self._compute_expectation_values(theta, X)
        ridge = Ridge(alpha=self.ridge_alpha).fit(ev, y)
        return float(np.mean((ridge.predict(ev) - y) ** 2))

    def _spsa_warmup(self, theta0, X, y, a=0.3, c=0.2, alpha_decay=0.602, gamma=0.101) -> np.ndarray:
        rng = np.random.RandomState(self.random_state)
        theta = theta0.copy()
        for k in range(1, self.maxiter_spsa + 1):
            ak = a / (k + 1) ** alpha_decay
            ck = c / (k ** gamma)
            delta = rng.choice([-1.0, 1.0], size=theta.shape)
            ev_plus, ev_minus = self._ev_batch_pair(theta + ck * delta, theta - ck * delta, X)
            loss_plus = float(np.mean((Ridge(alpha=self.ridge_alpha).fit(ev_plus, y).predict(ev_plus) - y) ** 2))
            loss_minus = float(np.mean((Ridge(alpha=self.ridge_alpha).fit(ev_minus, y).predict(ev_minus) - y) ** 2))
            ghat = (loss_plus - loss_minus) / (2 * ck * delta)
            theta = theta - ak * ghat
        return theta

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_MultiObservableRegressorCore":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self._qc, self._x_params, self._theta_params, self._output_qubits = self._build_circuit(X.shape[1])

        rng = np.random.RandomState(self.random_state)
        theta0 = rng.uniform(0, 2 * np.pi, size=len(self._theta_params))

        theta_warm = self._spsa_warmup(theta0, X, y)
        res = minimize(
            lambda th: self._mse_loss(th, X, y), theta_warm,
            method="COBYLA", options={"maxiter": self.maxiter_cobyla, "rhobeg": 0.3},
        )
        self._theta_opt = res.x

        ev_train = self._compute_expectation_values(self._theta_opt, X)
        self._ridge = Ridge(alpha=self.ridge_alpha)
        self._ridge.fit(ev_train, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        ev = self._compute_expectation_values(self._theta_opt, np.asarray(X, dtype=float))
        return self._ridge.predict(ev)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return r2_score(y, self.predict(X))


class VariationalQuantumRegressor(_MultiObservableRegressorCore):
    """VQR: angle encoding + `regression_ansatz` + multi-observable readout,
    predicting COMDR15 continuously instead of `VariationalQuantumClassifier`'s
    single-qubit P(|1>) in {0, 1}. See `_MultiObservableRegressorCore` for
    the shared fit/predict/optimization machinery.

    Parameters
    ----------
    n_layers : int, default=1
        `regression_ansatz` depth. Kept at 1 by default (21 params for
        n_qubits=6) rather than the more expressive `n_layers=2` (42
        params) to avoid over-parameterizing a 28-sample LOOCV training
        fold; raise deliberately if LOOCV shows clear underfitting.
    n_output_qubits : int, default=3
    maxiter_spsa, maxiter_cobyla, ridge_alpha, random_state :
        See `_MultiObservableRegressorCore`.
    """

    def __init__(self, n_layers: int = 1, n_output_qubits: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.n_layers = n_layers
        self.n_output_qubits = n_output_qubits

    def _build_circuit(self, n_features: int):
        qc, x, theta = build_regression_circuit(n_features, self.n_layers, self.n_output_qubits)
        return qc, x, theta, list(range(self.n_output_qubits))


class QCNNRegressor(_MultiObservableRegressorCore):
    """QCNN-R: the same 6-qubit conv+pool+dense-SU(4) architecture as
    `QCNNClassifier`, reused for regression by reading out BOTH surviving
    qubits' <Z> expectation values (instead of thresholding a single
    qubit's P(|1>) into a class) and combining them with a classical Ridge
    head. This gives the existing QCNN a direct regression counterpart,
    rounding out the regression suite to 3 models: QK-KRR, VQR, and QCNN-R.

    Requires exactly `n_qubits` (default 6) input features, same fixed-size
    constraint as `QCNNClassifier` -- run it on the dedicated 6-feature
    subset (`selection["features_by_k"][6]`), not whichever k the automated
    selector found best for the other models.
    """

    def __init__(self, n_qubits: int = 6, **kwargs):
        super().__init__(**kwargs)
        self.n_qubits = n_qubits

    def _build_circuit(self, n_features: int):
        if n_features != self.n_qubits:
            raise ValueError(f"QCNN-R requires exactly {self.n_qubits} input features, got {n_features}")
        qc, x, theta, final_qubits = build_qcnn_circuit(self.n_qubits)
        return qc, x, theta, final_qubits


# Each model builder takes the shared QuantumExecutor (resolved once per
# suite run) and returns a fresh, unfitted model. This is called once per
# LOOCV fold.
# Every variational model below gets its own random_state offset from the
# shared RANDOM_SEED rather than all defaulting to the same value. This
# doesn't change any single model's own LOOCV result (each was verified to
# already produce distinct per-sample predictions from the others even
# under a shared seed -- small-N aggregate metrics can coincide by chance,
# see the README/investigation notes), but sharing one RNG stream across
# architecturally different models is still bad practice for a comparison
# suite -- it needlessly correlates their initial parameter draws instead
# of letting each model's result be independent evidence.
QUANTUM_MODEL_BUILDERS = {
    "QK-SVM_angle": lambda executor: QuantumKernelSVM(executor=executor, feature_map_name="angle"),
    # maxiter is set lower here than the class default of 60. The trained
    # kernel model recomputes a full kernel matrix on every COBYLA
    # iteration, which costs far more real circuit executions per step
    # than the other variational models below. maxiter=30 gets most of the
    # benefit at a fraction of the runtime. For the full quality version
    # (96.6% accuracy with n_layers=3 and maxiter=60), construct
    # TrainedQuantumKernelSVM directly with those settings, the same way
    # the KTA demo in bonus_extensions.py does. See the README for the
    # comparison between different settings.
    "QK-SVM_trained": lambda executor: TrainedQuantumKernelSVM(
        executor=executor, n_layers=3, maxiter=30, random_state=RANDOM_SEED + 1
    ),
    "VQC": lambda executor: VariationalQuantumClassifier(
        executor=executor, n_layers=2, optimizer="cobyla", maxiter=60, random_state=RANDOM_SEED + 2
    ),
    "QCNN": lambda executor: QCNNClassifier(
        executor=executor, n_qubits=6, optimizer="cobyla", maxiter=80, random_state=RANDOM_SEED + 4
    ),
}

# Regression counterparts, kept in a separate registry (and run through
# `run_quantum_regression_suite` / `loocv_evaluate_regression`) since they
# predict COMDR15 continuously rather than the binary responder label that
# every entry in QUANTUM_MODEL_BUILDERS above targets. QCNN-R has the same
# fixed-6-qubit constraint as QCNN in QUANTUM_MODEL_BUILDERS -- callers
# should run it on the dedicated 6-feature subset, same as the classifier.
QUANTUM_REGRESSION_MODEL_BUILDERS = {
    # feature_map_name="angle", not zzplus (despite the class default and
    # the module docstring's original zzplus recommendation): measured
    # against the real 29-drug dataset (4-feature MANUAL_FEATURES set,
    # log-target, aer_simulator), zzplus/zz both severely overfit --
    # r2_train ~0.98-0.99 (near-perfect memorization) but q2_loocv < 0
    # (worse than predicting the training mean) at every reps in {1,2,3}.
    # This is the classic quantum-kernel-concentration failure mode
    # (Thanasilp et al. 2022; Kubler et al. 2021): ZZ cross-terms push the
    # induced Hilbert space's off-diagonal fidelities toward a small,
    # near-uniform value once qubit count/depth outgrows what 28 training
    # samples can constrain, and no amount of extra KernelRidge alpha
    # regularization recovers it (the inner alpha sweep kept picking the
    # *smallest* candidate even up to alpha_grid=(...,100.0), meaning more
    # shrinkage wasn't the fix -- the kernel itself carried too little
    # signal). Dropping to the plain angle encoding (no ZZ interactions)
    # measured q2_loocv=0.52 on the same data/settings, so it is the
    # default here instead. alpha_grid is kept wider than the class's own
    # default (0.01-1.0) as a safety margin.
    "QK-KRR_angle": lambda executor: QuantumKernelRidgeRegression(
        executor=executor, feature_map_name="angle", alpha=1.0, alpha_grid=(0.1, 1.0, 10.0, 100.0),
    ),
    # n_layers=1 is the optimal depth for this dataset's regime, not just a
    # cautious default: `regression_ansatz` has n_layers * n_qubits * 3 +
    # n_output_qubits parameters, so on a 4-qubit MANUAL_FEATURES set with
    # n_output_qubits=3, n_layers=1/2/3 give 15/27/39 trainable parameters
    # against only 28 LOOCV training samples per fold -- n_layers=2 is
    # already borderline (27 params ~ 28 samples) and n_layers=3 clearly
    # over-parameterized. Confirmed empirically at n_layers=1: a full real
    # 29-fold LOOCV run (log-target, aer_simulator, this exact maxiter
    # config) measured q2_loocv=0.60, r2_train=0.86 -- a healthy
    # generalization gap, not the collapse an over-parameterized fit would
    # show. n_layers=2/3 were not independently re-run under the same full
    # LOOCV budget (an exhaustive sweep like TrainedQuantumKernelSVM's
    # n_layers=1-5 classification study would cost several full 29-fold
    # LOOCV runs here), so this is the parameter-budget argument plus one
    # confirmed real result, not an exhaustive multi-value comparison.
    "VQR_spsa_cobyla": lambda executor: VariationalQuantumRegressor(
        executor=executor, n_layers=1, n_output_qubits=3, maxiter_spsa=50, maxiter_cobyla=20
    ),
    "QCNN-R": lambda executor: QCNNRegressor(
        executor=executor, n_qubits=6, maxiter_spsa=50, maxiter_cobyla=20
    ),
}


def run_quantum_classification_suite(
    X: np.ndarray,
    y: np.ndarray,
    execution_config: ExecutionConfig | None = None,
    model_names: list[str] | None = None,
) -> dict:
    """Run the requested quantum classifiers (all of them by default)
    through the shared LOOCV harness. Every model shares one
    QuantumExecutor, so the backend is only resolved once and the transpile
    pass manager is only built once, instead of once per fold. This
    matters most for IBM Runtime mode, where reconnecting per fold would
    be slow.
    """
    from src.evaluation import loocv_evaluate

    executor = QuantumExecutor(execution_config)
    names = model_names or list(QUANTUM_MODEL_BUILDERS.keys())
    results = {}
    for name in names:
        logger.info("Running quantum classifier: %s (backend=%s)", name, executor.config.label())
        factory = lambda name=name: QUANTUM_MODEL_BUILDERS[name](executor)
        # Angle encoding circuits use the raw feature value as a rotation
        # angle, so features need to be scaled to [0, pi] per fold, not
        # the [0, 1] range used for classical models. Without this every
        # rotation would collapse to a tiny sliver near the north pole of
        # the Bloch sphere and the circuit would barely distinguish inputs.
        results[name] = loocv_evaluate(factory, X, y, feature_range=(0.0, np.pi))
        logger.info("  %s -> accuracy=%.3f f1=%.3f (%.1fs)", name,
                     results[name]["metrics"]["accuracy"], results[name]["metrics"]["f1"],
                     results[name]["metrics"]["elapsed_s"])
    return results


def run_quantum_regression_suite(
    X: np.ndarray,
    y_continuous: np.ndarray,
    execution_config: ExecutionConfig | None = None,
    model_names: list[str] | None = None,
    log_target: bool = True,
) -> dict:
    """Run the requested quantum regressors (all of `QUANTUM_REGRESSION_MODEL_BUILDERS`
    by default) through the shared LOOCV harness, predicting COMDR15
    directly -- the regression counterpart of `run_quantum_classification_suite`.

    Every model shares one `QuantumExecutor`, for the same reason as the
    classification suite (a single backend resolution / transpile pass
    manager reused across every fold and model).

    Parameters
    ----------
    log_target : bool, default=True
        Fit and score every regressor against log(COMDR15) rather than the
        raw ratio, matching both the Patzmann et al. benchmark table
        (whose target row is literally "LogCOMDR15min") and this project's
        own `run_pls_regression_baseline` (also `log_target=True` by
        default). COMDR_15min spans a ~24x range and is heavily
        right-skewed by a handful of extreme high-responder outliers (e.g.
        Fenofibrate at 26.48), which dominates a raw-scale fit under only
        28-29 LOOCV training samples -- on the classical PLS baseline this
        alone moved Q^2 from ~0.45 to ~0.76. `r2`/`q2` are reported on
        that log scale (directly comparable to Patzmann's 0.82/0.77);
        `mae`/`rmse` are reported back on the original COMDR_15min scale
        (via `exp()`) since absolute error in log-ratio units is not
        physically interpretable.

    Each result's `metrics` dict has `r2` (full-data refit R^2, comparable
    to the Patzmann R^2=0.82 benchmark), `q2` (LOOCV R^2, comparable to
    their Q^2=0.77 benchmark -- see `loocv_evaluate_regression`'s
    docstring for why this is the same formula as `r2_score(y_true,
    y_pred)` on the LOOCV predictions), `mae`, `rmse`, `target_scale`, and
    `elapsed_s`.
    """
    from src.evaluation import loocv_evaluate_regression

    executor = QuantumExecutor(execution_config)
    names = model_names or list(QUANTUM_REGRESSION_MODEL_BUILDERS.keys())
    target = np.log(np.asarray(y_continuous, dtype=float)) if log_target else np.asarray(y_continuous, dtype=float)
    target_scale = "log(COMDR_15min)" if log_target else "COMDR_15min"
    results = {}
    for name in names:
        logger.info(
            "Running quantum regressor: %s (backend=%s, target=%s)", name, executor.config.label(), target_scale
        )
        factory = lambda name=name: QUANTUM_REGRESSION_MODEL_BUILDERS[name](executor)
        # Same [0, pi] angle-encoding rationale as the classification suite.
        raw = loocv_evaluate_regression(factory, X, target, feature_range=(0.0, np.pi))
        y_true, y_pred = raw["y_true"], raw["y_pred"]
        # r2/q2 stay on the (log-)scale the model was actually fit on, for
        # a direct comparison against the Patzmann benchmark; mae/rmse are
        # converted back to physical COMDR_15min units for interpretability.
        y_true_phys = np.exp(y_true) if log_target else y_true
        y_pred_phys = np.exp(y_pred) if log_target else y_pred
        mae = float(np.mean(np.abs(y_true_phys - y_pred_phys)))
        rmse = float(np.sqrt(np.mean((y_true_phys - y_pred_phys) ** 2)))

        results[name] = {
            "metrics": {
                "r2": raw["metrics"]["r2_train"],
                "q2": raw["metrics"]["q2_loocv"],
                "mae": mae,
                "rmse": rmse,
                "target_scale": target_scale,
                "elapsed_s": raw["metrics"]["elapsed_s"],
            },
            "predictions": y_pred_phys.tolist(),
            "targets": y_true_phys.tolist(),
            "scores": y_pred_phys.tolist(),
        }
        logger.info(
            "  %s -> q2=%.3f r2=%.3f mae=%.3f rmse=%.3f (%.1fs)", name,
            results[name]["metrics"]["q2"], results[name]["metrics"]["r2"], mae, rmse,
            results[name]["metrics"]["elapsed_s"],
        )
    return results
