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

import numpy as np
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


def zzplus_feature_map(
    n_features: int, reps: int = 3, full_entanglement: bool = True
) -> tuple[QuantumCircuit, ParameterVector]:
    """Extended ZZ feature map: a linear `ZZFeatureMap` plus two extra
    circular ZZ couplings closing the entanglement ring.

    Built on top of Qiskit's `ZZFeatureMap` (angle encoding + pairwise ZZ
    interaction terms between linear neighbours), then -- if
    `full_entanglement` and there are enough qubits -- two more pairwise ZZ
    interactions are appended manually: (0, n-2) and (1, n-1). These are
    added as a single CX-RZ-CX block per pair *after* the base map rather
    than by re-instantiating `ZZFeatureMap` with a different entanglement
    map, so no parameters are duplicated across `reps`. The appended phase
    uses the same convention as `ZZFeatureMap`'s own pairwise term (Qiskit's
    default `self_product` data map): phi(x_i, x_j) = (pi - x_i) * (pi - x_j).

    Physically, every ZZ term encodes a product x_i * x_j of two features --
    correlations that a pure angle encoding (independent Ry rotations)
    cannot represent. The two extra circular couplings reach qubit pairs
    that a purely linear entanglement map skips, giving the induced kernel
    more pairwise structure to work with without adding qubits or reps.

    Parameters
    ----------
    n_features : int
        Number of qubits / classical features encoded.
    reps : int, default=3
        Repetitions of the base `ZZFeatureMap` block.
    full_entanglement : bool, default=True
        If True and `n_features >= 4`, append the two circular ZZ couplings.

    Returns
    -------
    tuple[QuantumCircuit, ParameterVector]
        The bound circuit and its length-`n_features` data ParameterVector.
    """
    x = ParameterVector("x", n_features)
    # Build the base ZZFeatureMap on its own parameters, then rebind those
    # parameters to `x` so the extra couplings below share the same
    # ParameterVector instance instead of introducing a second one.
    base = ZZFeatureMap(feature_dimension=n_features, reps=reps, entanglement="linear")
    bound_base = base.assign_parameters(dict(zip(base.parameters, x)))

    qc = QuantumCircuit(n_features, name="ZZPlusFeatureMap")
    qc.compose(bound_base, inplace=True)

    if full_entanglement and n_features >= 4:
        circular_pairs = [(0, n_features - 2), (1, n_features - 1)]
        for i, j in circular_pairs:
            if i == j:
                continue
            qc.cx(i, j)
            qc.rz(2.0 * (np.pi - x[i]) * (np.pi - x[j]), j)
            qc.cx(i, j)
    return qc, x


def regression_ansatz(
    n_qubits: int, n_layers: int, n_output_qubits: int = 3
) -> tuple[QuantumCircuit, ParameterVector]:
    """Trainable ansatz for continuous-output regression: Ry/Rz/Ry rotation
    layers (three rotations per qubit per layer, more expressive than the
    VQC's Ry/Rz ansatz) with circular CNOT entanglement, ending in a
    Ry-only layer applied only to the first `n_output_qubits` qubits.

    That final partial layer exists to maximize the variance of the
    <Z_i> expectation values that will actually be read out (see
    `VariationalQuantumRegressor`) -- qubits that are never read out don't
    need a dedicated final rotation, only enough entanglement to have
    already influenced the output qubits' state.

    Total trainable parameters: `n_layers * n_qubits * 3 + n_output_qubits`.

    Parameters
    ----------
    n_qubits : int
    n_layers : int
        Depth knob. With `n_qubits=6`, `n_layers=2` gives 42 parameters --
        close to overparameterizing a 28-sample LOOCV training fold, so
        callers should default to `n_layers=1` (21 params) and only raise
        it if LOOCV shows clear underfitting.
    n_output_qubits : int, default=3
        Number of qubits whose <Z> is intended to be measured downstream.

    Returns
    -------
    tuple[QuantumCircuit, ParameterVector]
    """
    if n_output_qubits > n_qubits:
        raise ValueError(f"n_output_qubits ({n_output_qubits}) cannot exceed n_qubits ({n_qubits})")

    theta = ParameterVector("theta", n_layers * n_qubits * 3 + n_output_qubits)
    qc = QuantumCircuit(n_qubits, name="RegressionAnsatz")
    idx = 0
    for _layer in range(n_layers):
        for q in range(n_qubits):
            qc.ry(theta[idx], q)
            idx += 1
            qc.rz(theta[idx], q)
            idx += 1
            qc.ry(theta[idx], q)
            idx += 1
        if n_qubits > 1:
            for q in range(n_qubits - 1):
                qc.cx(q, q + 1)
            qc.cx(n_qubits - 1, 0)
    for q in range(n_output_qubits):
        qc.ry(theta[idx], q)
        idx += 1
    return qc, theta


def build_regression_circuit(
    n_features: int, n_layers: int = 1, n_output_qubits: int = 3
) -> tuple[QuantumCircuit, ParameterVector, ParameterVector]:
    """Full VQR circuit: the angle encoding feature map followed by `regression_ansatz`."""
    fm, x = angle_feature_map(n_features, entangle=True)
    ansatz, theta = regression_ansatz(n_features, n_layers, n_output_qubits)
    qc = QuantumCircuit(n_features, name="RegressionCircuit")
    qc.compose(fm, inplace=True)
    qc.compose(ansatz, inplace=True)
    return qc, x, theta
