"""Circuit builders for the quantum feature maps and ansatze used across the project.

This file only defines circuit shapes. Nothing in here runs a circuit or
touches any data, that happens in quantum_backend.py and quantum_models.py.
Three feature map styles are provided so their expressibility can be
compared (this is Task 3 of the challenge):

- angle_feature_map: one Ry rotation per qubit, then an optional CNOT chain.
  This is the baseline encoding described in the challenge brief.
- entangled_feature_map: the angle encoding repeated a few times with a
  CNOT ring after each repetition, to see if extra entanglement alone
  helps without adding any trainable parameters.
- zz_feature_map: Qiskit's built in ZZFeatureMap, which adds pairwise
  interaction terms between qubits on top of angle encoding.

reuploading_layer and variational_ansatz build the trainable circuits used
by the data re-uploading classifier, the VQC, and the trained quantum
kernel. Every builder function returns the circuit plus its parameter
vectors, so the caller can bind data values and, where relevant, trainable
weights separately.
"""

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZZFeatureMap


def angle_feature_map(n_features: int, entangle: bool = True) -> tuple[QuantumCircuit, ParameterVector]:
    """One Ry rotation per qubit, then an optional CNOT chain linking neighboring qubits."""
    x = ParameterVector("x", n_features)
    qc = QuantumCircuit(n_features, name="AngleFeatureMap")
    for i in range(n_features):
        qc.ry(x[i], i)
    if entangle:
        for i in range(n_features - 1):
            qc.cx(i, i + 1)
    return qc, x


def entangled_feature_map(n_features: int, reps: int = 2) -> tuple[QuantumCircuit, ParameterVector]:
    """Angle encoding repeated reps times, with a CNOT ring closing after each repetition.

    This has more entanglement than angle_feature_map but still has no
    trainable weights, so it is a way to check whether entanglement by
    itself helps Kernel Target Alignment.
    """
    x = ParameterVector("x", n_features)
    qc = QuantumCircuit(n_features, name="EntangledFeatureMap")
    for _ in range(reps):
        for i in range(n_features):
            qc.ry(x[i], i)
        for i in range(n_features - 1):
            qc.cx(i, i + 1)
        qc.cx(n_features - 1, 0)  # close the ring back to qubit 0
    return qc, x


def zz_feature_map(n_features: int, reps: int = 2) -> tuple[QuantumCircuit, ParameterVector]:
    """Qiskit's ZZFeatureMap: angle encoding plus pairwise ZZ interaction terms."""
    fm = ZZFeatureMap(feature_dimension=n_features, reps=reps, entanglement="linear")
    fm.name = "ZZFeatureMap"
    params = ParameterVector("x", n_features)
    bound = fm.assign_parameters(dict(zip(fm.parameters, params)))
    return bound, params


def reuploading_layer(
    n_features: int,
    n_layers: int,
) -> tuple[QuantumCircuit, ParameterVector, ParameterVector]:
    """Data re-uploading block: a trainable rotation, then the data encoding,
    then entanglement, repeated n_layers times. The same qubits see the
    data more than once, which gives the circuit more expressive power
    without needing more qubits.

    Returns the circuit, the feature parameters (n_features of them, bound
    the same way at every layer), and the weight parameters
    (n_layers * n_features * 2 of them: one trainable Ry and one Rz per
    qubit per layer).
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
    """Trainable ansatz used by the VQC: Ry and Rz rotations followed by a
    CNOT chain, repeated n_layers times.
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
    """Full VQC circuit: the angle encoding feature map followed by the trainable ansatz."""
    fm, x = angle_feature_map(n_features, entangle=True)
    ansatz, theta = variational_ansatz(n_features, n_layers)
    qc = QuantumCircuit(n_features, name="VQC")
    qc.compose(fm, inplace=True)
    qc.compose(ansatz, inplace=True)
    return qc, x, theta
