"""Executable quantum ML model classes: QK-SVM, VQC, Data Re-uploading, QCNN.

Every probability and kernel entry used by these models is obtained from an
*actual* `SamplerV2` circuit execution (see `quantum_backend.py`) -- shots
sampled from a transpiled circuit run on either a local `AerSimulator` or a
real IBM Quantum backend, never a shortcut `Statevector` linear-algebra
calculation. All circuits needed for one LOOCV fold (or one optimizer
iteration) are batched into a single Sampler job wherever possible, since
that is what makes real-hardware/cloud-queue execution practical.
"""

import logging
import warnings

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from scipy.optimize import minimize
from sklearn.svm import SVC

from src.config import RANDOM_SEED
from src.quantum_backend import (
    ExecutionConfig,
    QuantumExecutor,
    build_compute_uncompute_circuit,
    build_measurement_circuit,
    default_execution_config,
    fidelity_from_counts,
    probability_of_one,
)
from src.quantum_circuits import (
    angle_feature_map,
    build_vqc_circuit,
    entangled_feature_map,
    reuploading_layer,
    zz_feature_map,
)

logger = logging.getLogger(__name__)

# SVC(kernel="precomputed", probability=True) is deprecated in favor of
# CalibratedClassifierCV as of sklearn 1.9, but CalibratedClassifierCV's
# internal CV splitting of a precomputed kernel matrix is fragile on
# 28-sample training folds; probability=True remains fully functional
# (removal not until sklearn 1.11), so the warning is suppressed deliberately
# here rather than swapped for a riskier untested path.
warnings.filterwarnings("ignore", message=".*`probability` parameter was deprecated.*", category=FutureWarning)
# scipy's COBYLA silently clamps maxiter up to num_vars+2 internally and
# warns every time it does -- expected/harmless here (small ansatzes
# legitimately need few iterations), but deafening across a 29-fold LOOCV.
warnings.filterwarnings("ignore", message=".*Invalid MAXFUN.*", category=UserWarning)

# Registry of available feature maps -- add a new encoding here and every
# model/script that iterates FEATURE_MAPS picks it up automatically.
FEATURE_MAPS = {
    "angle": angle_feature_map,
    "entangled": entangled_feature_map,
    "zz": zz_feature_map,
}


# --- Real-execution kernel utilities ----------------------------------------

def _fidelity_kernel_via_backend(
    executor: QuantumExecutor,
    qc: QuantumCircuit,
    x_params: ParameterVector,
    X_a: np.ndarray,
    X_b: np.ndarray,
    symmetric: bool,
) -> np.ndarray:
    """NxM fidelity kernel matrix, every entry measured via an actual
    compute-uncompute circuit run through `executor`. When `symmetric` (X_a
    is X_b, e.g. the training kernel), only the upper triangle + diagonal is
    submitted -- exact fidelity is symmetric by construction -- roughly
    halving the number of real circuit executions needed.
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
    """Standalone NxN kernel matrix for a feature map -- used for the Task 3
    kernel heatmap / KTA deliverable, and reusable anywhere a raw quantum
    kernel is needed independent of a trained SVM.
    """
    qc, x_params = FEATURE_MAPS[feature_map_name](X.shape[1], **feature_map_kwargs)
    return _fidelity_kernel_via_backend(executor, qc, x_params, X, X, symmetric=True)


def _bce_loss(probs: np.ndarray, y: np.ndarray) -> float:
    eps = 1e-9
    p = np.clip(probs, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


# --- Generic variational training core --------------------------------------

class _VariationalCore:
    """Shared fit/predict machinery for VQC, Data Re-uploading, and QCNN.

    Subclasses implement `_build_circuit(n_features)` and set `self.output_qubit`.
    Requires a live `QuantumExecutor` (shared across an entire LOOCV run, not
    recreated per fold, so an IBM Runtime backend is resolved only once).

    Optimizer choices trade off job count vs. gradient fidelity -- every
    option below batches as many circuits as possible into as few Sampler
    jobs as possible, since job count (not circuit count) dominates wall
    time on a real queued backend:
      - "cobyla"          : gradient-free; one Sampler job (N circuits, N =
                             training-fold size) per scalar-loss evaluation.
      - "spsa"             : one Sampler job of 2N circuits per iteration
                              (the +/- perturbation pair batched together),
                              regardless of parameter count -- the standard
                              choice for real-hardware variational training.
      - "parameter_shift"  : exact analytic gradient; one Sampler job of
                              2*n_params*N circuits per iteration. Most
                              accurate but most expensive -- intended for
                              small illustrative comparisons, not full LOOCV.
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
        """Both perturbed-parameter evaluations submitted as ONE Sampler job."""
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

            # Batch every +/-shift evaluation for every parameter into one job.
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
    """QK-SVM: precomputed fidelity-kernel matrix (measured via real circuit
    execution) + a classical SVM head.
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


# --- 1b. Trained (Kernel-Target-Alignment-optimized) Quantum Kernel SVM -----

class TrainedQuantumKernelSVM:
    """A quantum kernel whose feature map carries trainable weight
    parameters *in addition to* the data-encoding parameters -- optimized
    via COBYLA to maximize Kernel Target Alignment (KTA) against the
    training labels before the classical SVM head is fit. This is a
    Havlicek-style *trained* quantum kernel, in contrast to
    `QuantumKernelSVM`'s fixed encoding.

    Why this class exists (see README for the full investigation): a fixed
    encoding's raw KTA on this dataset is unremarkable (~0.2-0.3) -- not
    because the encoding is broken, but because closed-form KTA computed
    on all 29 samples at once is *not*, by itself, a reliable proxy for
    downstream generalization. Empirically: widening the fixed encoding's
    rotation range pushes closed-loop KTA up while making fold-safe LOOCV
    accuracy *worse* (93%->83%) -- textbook overfitting to a metric rather
    than the task. What actually, honestly helps is training the kernel's
    own parameters against KTA computed strictly within each fold's
    training data (never the held-out sample) via `reuploading_layer`'s
    data re-uploading structure. `n_layers` was swept 1-5 under full
    fold-safe LOOCV: 1-2 give ~93% accuracy (matching the fixed encoding),
    3 peaks at ~97%, and 4-5 overfit and *drop* below the fixed baseline
    even though their raw KTA keeps climbing -- confirming KTA must be
    validated fold-safe, not chased directly.
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
        logger.info("Trained-kernel KTA (this fold): %.4f -> %.4f", self.kta_before_, self.kta_after_)
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
    """Angle encoding + trainable Ry/Rz + CNOT-chain ansatz, BCE-trained."""

    def __init__(self, n_layers: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.n_layers = n_layers

    def _build_circuit(self, n_features: int):
        self.output_qubit = 0
        return build_vqc_circuit(n_features, self.n_layers)


# --- 3. Data Re-uploading Classifier -----------------------------------------

class DataReuploadingClassifier(_VariationalCore):
    """Repeats [trainable rotation, data re-encoding, entanglement] `n_layers`
    times -- universal approximation power without extra qubits.
    """

    def __init__(self, n_layers: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.n_layers = n_layers

    def _build_circuit(self, n_features: int):
        self.output_qubit = 0
        return reuploading_layer(n_features, self.n_layers)


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
    """Generic SU(4) dense layer on 2 qubits (15 free real parameters --
    the full dimension of su(4)) capturing all pairwise correlations
    between the two surviving qubits.
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


def build_qcnn_circuit(n_qubits: int = 6) -> tuple[QuantumCircuit, ParameterVector, ParameterVector, int]:
    """6-qubit angle-encoded QCNN: conv+pool (6->3), conv+pool (3->2), SU(4)
    dense layer on the 2 surviving qubits, single output qubit.
    """
    x = ParameterVector("x", n_qubits)
    qc = QuantumCircuit(n_qubits, name="QCNN")
    for i in range(n_qubits):
        qc.ry(x[i], i)

    n_conv1 = n_qubits          # circular pairs (0,1),(1,2),...,(n-1,0)
    n_pool1 = n_qubits // 2     # 6 -> 3
    n_conv2 = 3                 # circular pairs among the 3 survivors
    n_pool2 = 1                 # 3 -> 2 (one pair pooled, one qubit passes through)
    n_dense = 15                # full SU(4) block on the final 2 qubits

    theta = ParameterVector("theta", 3 * (n_conv1 + n_pool1 + n_conv2 + n_pool2) + n_dense)
    idx = 0

    # Conv layer 1 on all 6 qubits (circular neighbours).
    for i in range(n_qubits):
        j = (i + 1) % n_qubits
        qc.compose(_conv_block(theta[idx:idx + 3]), qubits=[i, j], inplace=True)
        idx += 3

    # Pool layer 1: 6 -> 3, keep qubits {1, 3, 5}.
    pool1_pairs = [(0, 1), (2, 3), (4, 5)]
    for src, keep in pool1_pairs:
        qc.compose(_pool_block(theta[idx:idx + 3]), qubits=[src, keep], inplace=True)
        idx += 3
    survivors1 = [1, 3, 5]

    # Conv layer 2 on the 3 survivors (circular neighbours).
    for a, b in [(survivors1[0], survivors1[1]), (survivors1[1], survivors1[2]), (survivors1[2], survivors1[0])]:
        qc.compose(_conv_block(theta[idx:idx + 3]), qubits=[a, b], inplace=True)
        idx += 3

    # Pool layer 2: 3 -> 2, pool (survivors1[0], survivors1[1]) into survivors1[1];
    # survivors1[2] passes through untouched.
    qc.compose(_pool_block(theta[idx:idx + 3]), qubits=[survivors1[0], survivors1[1]], inplace=True)
    idx += 3
    final_qubits = [survivors1[1], survivors1[2]]

    # Dense SU(4) layer on the final 2 qubits.
    qc.compose(_dense_su4_block(theta[idx:idx + n_dense]), qubits=final_qubits, inplace=True)
    idx += n_dense

    output_qubit = final_qubits[0]
    return qc, x, theta, output_qubit


class QCNNClassifier(_VariationalCore):
    """6-to-1 qubit Quantum Convolutional Neural Network."""

    def __init__(self, n_qubits: int = 6, **kwargs):
        super().__init__(**kwargs)
        self.n_qubits = n_qubits

    def _build_circuit(self, n_features: int):
        if n_features != self.n_qubits:
            raise ValueError(f"QCNN requires exactly {self.n_qubits} input features, got {n_features}")
        qc, x, theta, output_qubit = build_qcnn_circuit(self.n_qubits)
        self.output_qubit = output_qubit
        return qc, x, theta


# Model builders take a shared QuantumExecutor (resolved once per suite run)
# and return a fresh, unfitted model instance -- called once per LOOCV fold.
QUANTUM_MODEL_BUILDERS = {
    "QK-SVM_angle": lambda executor: QuantumKernelSVM(executor=executor, feature_map_name="angle"),
    "QK-SVM_zz": lambda executor: QuantumKernelSVM(executor=executor, feature_map_name="zz"),
    # maxiter=30 here (not the class's own default of 60) is a deliberate
    # real-execution cost tradeoff: unlike the O(N) per-iteration cost of
    # the _VariationalCore models below, each COBYLA iteration here
    # recomputes an O(N^2) kernel matrix (~406 real circuits for a 28-sample
    # training fold), so this single model is ~14x more expensive per
    # optimizer step. Empirically (fast exact-statevector sweep, see
    # TrainedQuantumKernelSVM's docstring), maxiter=26-30 (COBYLA silently
    # floors maxiter at num_vars+2=26 for this 24-param search regardless of
    # what's requested below that) already converges to the same fold-safe
    # LOOCV accuracy (~90%) as maxiter=60 without training beyond that floor
    # -- the extra iterations up to 60 only pay off in combination with
    # n_layers=3 across the *whole* dataset's specific optimization
    # landscape (~97%), and even then cost ~11h for a full 29-fold
    # real-execution LOOCV run. Construct TrainedQuantumKernelSVM(maxiter=60)
    # directly (as bonus_extensions.py's single-fit KTA demo does) if you
    # want that quality and can afford the wait.
    "QK-SVM_trained": lambda executor: TrainedQuantumKernelSVM(executor=executor, n_layers=3, maxiter=30),
    "VQC": lambda executor: VariationalQuantumClassifier(executor=executor, n_layers=2, optimizer="cobyla", maxiter=60),
    "DataReuploading": lambda executor: DataReuploadingClassifier(
        executor=executor, n_layers=3, optimizer="cobyla", maxiter=60
    ),
    "QCNN": lambda executor: QCNNClassifier(executor=executor, n_qubits=6, optimizer="cobyla", maxiter=80),
}


def run_quantum_classification_suite(
    X: np.ndarray,
    y: np.ndarray,
    execution_config: ExecutionConfig | None = None,
    model_names: list[str] | None = None,
) -> dict:
    """Run the requested quantum classifiers (default: all registered) through
    the shared LOOCV harness, all sharing ONE `QuantumExecutor` (one backend
    resolution, one transpile pass manager) across every fold and every
    model -- critical for IBM Runtime mode, where re-resolving a backend or
    re-building a pass manager per fold would be wasteful.
    """
    from src.evaluation import loocv_evaluate

    executor = QuantumExecutor(execution_config)
    names = model_names or list(QUANTUM_MODEL_BUILDERS.keys())
    results = {}
    for name in names:
        logger.info("Running quantum classifier: %s (backend=%s)", name, executor.config.label())
        factory = lambda name=name: QUANTUM_MODEL_BUILDERS[name](executor)
        # Angle-encoding circuits consume raw rotation angles (Ry(x_i)), so
        # features must be scaled to [0, pi] per fold -- NOT the [0, 1]
        # default used for classical models -- or every rotation collapses
        # into a useless sliver near the Bloch sphere's north pole.
        results[name] = loocv_evaluate(factory, X, y, feature_range=(0.0, np.pi))
        logger.info("  %s -> accuracy=%.3f f1=%.3f (%.1fs)", name,
                     results[name]["metrics"]["accuracy"], results[name]["metrics"]["f1"],
                     results[name]["metrics"]["elapsed_s"])
    return results
