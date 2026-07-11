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
    """Shared fit and predict logic for the VQC, the data re-uploading
    classifier, and the QCNN. These three models differ only in circuit
    shape, so all the training code lives here once.

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


# --- 3. Data Re-uploading Classifier -----------------------------------------

class DataReuploadingClassifier(_VariationalCore):
    """Repeats a trainable rotation, data re-encoding, and entanglement
    block n_layers times. Uploading the data more than once gives the
    circuit more expressive power without needing more qubits.
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


def build_qcnn_circuit(n_qubits: int = 6) -> tuple[QuantumCircuit, ParameterVector, ParameterVector, int]:
    """Build the QCNN circuit: 6 qubits with angle encoding, a conv and
    pool stage taking it from 6 qubits down to 3, a second conv and pool
    stage taking it from 3 down to 2, then a dense SU(4) layer on the
    final 2 qubits with a single output qubit.
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

    output_qubit = final_qubits[0]
    return qc, x, theta, output_qubit


class QCNNClassifier(_VariationalCore):
    """Quantum convolutional neural network. Takes 6 input qubits down to 1 output qubit."""

    def __init__(self, n_qubits: int = 6, **kwargs):
        super().__init__(**kwargs)
        self.n_qubits = n_qubits

    def _build_circuit(self, n_features: int):
        if n_features != self.n_qubits:
            raise ValueError(f"QCNN requires exactly {self.n_qubits} input features, got {n_features}")
        qc, x, theta, output_qubit = build_qcnn_circuit(self.n_qubits)
        self.output_qubit = output_qubit
        return qc, x, theta


# Each model builder takes the shared QuantumExecutor (resolved once per
# suite run) and returns a fresh, unfitted model. This is called once per
# LOOCV fold.
QUANTUM_MODEL_BUILDERS = {
    "QK-SVM_angle": lambda executor: QuantumKernelSVM(executor=executor, feature_map_name="angle"),
    "QK-SVM_zz": lambda executor: QuantumKernelSVM(executor=executor, feature_map_name="zz"),
    # maxiter is set lower here than the class default of 60. The trained
    # kernel model recomputes a full kernel matrix on every COBYLA
    # iteration, which costs far more real circuit executions per step
    # than the other variational models below. maxiter=30 gets most of the
    # benefit at a fraction of the runtime. For the full quality version
    # (96.6% accuracy with n_layers=3 and maxiter=60), construct
    # TrainedQuantumKernelSVM directly with those settings, the same way
    # the KTA demo in bonus_extensions.py does. See the README for the
    # comparison between different settings.
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
