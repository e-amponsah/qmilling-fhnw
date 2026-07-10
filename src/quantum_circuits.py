"""Parameterized quantum feature-map circuit layouts.

Three encoding variants are provided so expressibility/accuracy trade-offs
can be compared directly (Task 3 of the challenge):

- `angle_feature_map`      : one Ry(x_i) per qubit, optional single CNOT chain.
- `reuploading_feature_map`: the encoding block repeated `reps` times, each
                              repetition interleaved with a trainable Ry/Rz
                              layer -- the building block for both the
                              entangled feature map and the data re-uploading
                              classifier.
- `zz_feature_map`         : thin wrapper around Qiskit's ZZFeatureMap, which
                              adds pairwise ZZ-interaction encoding on top of
                              angle encoding.

Every builder returns `(circuit, feature_params, weight_params)` where
`weight_params` is an empty ParameterVector for purely-encoding maps.
"""

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZZFeatureMap


def angle_feature_map(n_features: int, entangle: bool = True) -> tuple[QuantumCircuit, ParameterVector]:
    """Ry(x_i) on each qubit, then an optional CNOT chain entangling neighbours."""
    x = ParameterVector("x", n_features)
    qc = QuantumCircuit(n_features, name="AngleFeatureMap")
    for i in range(n_features):
        qc.ry(x[i], i)
    if entangle:
        for i in range(n_features - 1):
            qc.cx(i, i + 1)
    return qc, x


def entangled_feature_map(n_features: int, reps: int = 2) -> tuple[QuantumCircuit, ParameterVector]:
    """Angle encoding repeated `reps` times with a CNOT chain after each repetition.

    Deepens the entanglement structure relative to `angle_feature_map` without
    introducing trainable weights -- useful for probing whether extra
    entanglement alone improves Kernel Target Alignment (Task 3).
    """
    x = ParameterVector("x", n_features)
    qc = QuantumCircuit(n_features, name="EntangledFeatureMap")
    for _ in range(reps):
        for i in range(n_features):
            qc.ry(x[i], i)
        for i in range(n_features - 1):
            qc.cx(i, i + 1)
        qc.cx(n_features - 1, 0)  # close the entangling ring
    return qc, x


def zz_feature_map(n_features: int, reps: int = 2) -> tuple[QuantumCircuit, ParameterVector]:
    """Qiskit's ZZFeatureMap: angle encoding + pairwise ZZ interaction terms."""
    fm = ZZFeatureMap(feature_dimension=n_features, reps=reps, entanglement="linear")
    fm.name = "ZZFeatureMap"
    params = ParameterVector("x", n_features)
    bound = fm.assign_parameters(dict(zip(fm.parameters, params)))
    return bound, params


def reuploading_layer(
    n_features: int,
    n_layers: int,
) -> tuple[QuantumCircuit, ParameterVector, ParameterVector]:
    """Data re-uploading block: repeats [trainable rotation, data encoding,
    entanglement] `n_layers` times so the same qubits see the data multiple
    times, giving universal-approximation power without extra qubits.

    Returns (circuit, feature_params, weight_params). `feature_params` has
    shape n_features (re-bound identically at every layer); `weight_params`
    has shape n_layers * n_features * 2 (a trainable Ry, Rz pair per qubit
    per layer).
    """
    x = ParameterVector("x", n_features)
    theta = ParameterVector("theta", n_layers * n_features * 2)
    qc = QuantumCircuit(n_features, name="DataReuploading")

    idx = 0
    for layer in range(n_layers):
        for q in range(n_features):
            qc.ry(theta[idx], q)
            idx += 1
            qc.rz(theta[idx], q)
            idx += 1
        for q in range(n_features):
            qc.ry(x[q], q)
        if n_features > 1:
            for q in range(n_features - 1):
                qc.cx(q, q + 1)
            qc.cx(n_features - 1, 0)
    return qc, x, theta


def variational_ansatz(n_qubits: int, n_layers: int) -> tuple[QuantumCircuit, ParameterVector]:
    """Trainable ansatz for the VQC: alternating Ry/Rz rotation layers and a
    CNOT-chain entangling layer, repeated `n_layers` times.
    """
    theta = ParameterVector("theta", n_layers * n_qubits * 2)
    qc = QuantumCircuit(n_qubits, name="VariationalAnsatz")
    idx = 0
    for layer in range(n_layers):
        for q in range(n_qubits):
            qc.ry(theta[idx], q)
            idx += 1
            qc.rz(theta[idx], q)
            idx += 1
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
    return qc, theta


def build_vqc_circuit(n_features: int, n_layers: int = 2) -> tuple[QuantumCircuit, ParameterVector, ParameterVector]:
    """Full VQC circuit = angle encoding feature map + trainable ansatz."""
    fm, x = angle_feature_map(n_features, entangle=True)
    ansatz, theta = variational_ansatz(n_features, n_layers)
    qc = QuantumCircuit(n_features, name="VQC")
    qc.compose(fm, inplace=True)
    qc.compose(ansatz, inplace=True)
    return qc, x, theta
