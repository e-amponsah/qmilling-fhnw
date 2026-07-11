"""This is the layer that actually runs quantum circuits.

Every quantum computation in this project, whether it is a kernel fidelity
or a variational model's prediction, goes through a real SamplerV2 job:
transpiled to the target backend and run with a finite number of shots.
Nothing in this project computes an exact result with linear algebra as a
shortcut. The only difference between running locally and running on IBM
hardware is which backend object the Sampler points at.

Pick the target with ExecutionConfig.mode:
    "aer_simulator"  local, noiseless AerSimulator. This is the default,
                     and it is fast and free.
    "aer_noisy"      local AerSimulator loaded with a noise model that
                     approximates a real device.
    "ibm_runtime"    a real (or cloud hosted) IBM backend, reached through
                     QiskitRuntimeService using the account saved by
                     scripts/setup_ibm_account.py.

Every quantum model class in quantum_models.py takes an ExecutionConfig and
does not otherwise need to know where its circuits are actually running.
"""

import logging
import os
from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

from src.config import RANDOM_SEED

logger = logging.getLogger(__name__)


def build_device_like_noise_model(
    single_qubit_error: float = 1e-3, two_qubit_error: float = 1e-2, readout_error: float = 0.02
) -> NoiseModel:
    """Build a noise model that approximates a real superconducting device:
    depolarizing error on single and two qubit gates plus a symmetric
    readout error, at levels typical of current hardware calibration
    reports. Used for mode="aer_noisy" when there is no live IBM backend
    configured.
    """
    noise_model = NoiseModel(basis_gates=["u", "cx"])
    noise_model.add_all_qubit_quantum_error(depolarizing_error(single_qubit_error, 1), ["u"])
    noise_model.add_all_qubit_quantum_error(depolarizing_error(two_qubit_error, 2), ["cx"])
    ro_error = ReadoutError([[1 - readout_error, readout_error], [readout_error, 1 - readout_error]])
    noise_model.add_all_qubit_readout_error(ro_error)
    return noise_model


@dataclass
class ExecutionConfig:
    """Everything needed to point a batch of circuits at a real backend."""

    mode: str = "aer_simulator"  # one of: aer_simulator, aer_noisy, ibm_runtime
    shots: int = 4096
    optimization_level: int = 1
    seed: int = RANDOM_SEED
    ibm_backend_name: str | None = None  # if not set, picks the least busy backend
    noise_model: NoiseModel | None = None  # override for aer_noisy, otherwise built automatically

    def label(self) -> str:
        if self.mode == "ibm_runtime":
            return f"ibm_runtime[{self.ibm_backend_name or 'least_busy'}]"
        return self.mode


def default_execution_config() -> ExecutionConfig:
    """Build an ExecutionConfig from environment variables (QC_BACKEND_MODE,
    QC_SHOTS, QC_IBM_BACKEND), so the execution target can be changed from
    .env or the shell without touching any code.
    """
    mode = os.environ.get("QC_BACKEND_MODE", "aer_simulator")
    shots = int(os.environ.get("QC_SHOTS", "4096"))
    ibm_backend_name = os.environ.get("QC_IBM_BACKEND") or None
    return ExecutionConfig(mode=mode, shots=shots, ibm_backend_name=ibm_backend_name)


class QuantumExecutor:
    """Resolves an ExecutionConfig to a live backend and Sampler once, then
    runs batches of circuits against it. One instance should be reused for
    an entire LOOCV run, or at least a full fold, instead of being
    recreated for every call. Recreating it would mean reconnecting to
    IBM Runtime and picking a new least busy backend every time, which is
    both slow and unnecessary.
    """

    def __init__(self, config: ExecutionConfig | None = None):
        self.config = config or default_execution_config()
        self.backend = self._resolve_backend()
        self.sampler = SamplerV2(mode=self.backend)
        self._pm = generate_preset_pass_manager(
            backend=self.backend, optimization_level=self.config.optimization_level, seed_transpiler=self.config.seed
        )
        logger.info("QuantumExecutor ready | backend=%s | shots=%d", self.config.label(), self.config.shots)

    def _resolve_backend(self):
        if self.config.mode == "aer_simulator":
            return AerSimulator(seed_simulator=self.config.seed)
        if self.config.mode == "aer_noisy":
            noise_model = self.config.noise_model or build_device_like_noise_model()
            return AerSimulator(noise_model=noise_model, seed_simulator=self.config.seed)
        if self.config.mode == "ibm_runtime":
            service = QiskitRuntimeService()
            if self.config.ibm_backend_name:
                return service.backend(self.config.ibm_backend_name)
            backend = service.least_busy(operational=True, simulator=False)
            logger.info("IBM Runtime: auto-selected least-busy backend '%s'", backend.name)
            return backend
        raise ValueError(f"Unknown ExecutionConfig.mode: {self.config.mode}")

    def run_counts_batch(self, circuits: list[QuantumCircuit]) -> list[dict[str, int]]:
        """Transpile and submit every circuit in one Sampler job, then
        return a list of measurement count dictionaries in the same order
        as the input circuits.

        Batching all the circuits needed for one LOOCV fold, or one full
        kernel matrix, into a single job is what makes running on a real
        or cloud queued backend practical. Submitting one job per circuit
        would be far too slow.
        """
        if not circuits:
            return []
        transpiled = self._pm.run(circuits)
        job = self.sampler.run(transpiled, shots=self.config.shots)
        result = job.result()

        counts_list = []
        for pub_result in result:
            data_bin = pub_result.data
            creg_name = next(iter(data_bin.keys())) if hasattr(data_bin, "keys") else "meas"
            bit_array = getattr(data_bin, creg_name)
            counts_list.append(bit_array.get_counts())
        return counts_list


def probability_of_one(counts: dict[str, int], qubit: int, shots: int) -> float:
    """Estimate P(qubit == 1) from a Sampler counts dictionary. Qiskit
    bitstrings are little endian, so the rightmost character is qubit 0.
    """
    ones = sum(c for bitstring, c in counts.items() if bitstring[::-1][qubit] == "1")
    return ones / shots


def expectation_value_z(counts: dict[str, int], qubit: int, shots: int) -> float:
    """Estimate <Z_q> = P(0) - P(1) for qubit q from a Sampler counts
    dictionary. Qiskit bitstrings are little endian, so the rightmost
    character is qubit 0.

    The result lies in [-1, 1] and is centered on 0, which is a better
    regression feature than raw P(1): it keeps a classical Ridge head from
    seeing every output qubit's signal pre-biased toward one end of [0, 1]
    (used by `VariationalQuantumRegressor` and `QCNNRegressor`).
    """
    if shots == 0 or not counts:
        return 0.0
    zeros = sum(c for bitstring, c in counts.items() if bitstring[::-1][qubit] == "0")
    ones = sum(c for bitstring, c in counts.items() if bitstring[::-1][qubit] == "1")
    return (zeros - ones) / shots


def fidelity_from_counts(counts: dict[str, int], num_qubits: int, shots: int) -> float:
    """Estimate fidelity from a compute-uncompute circuit's measurement
    counts: it is the probability of measuring all zeros.
    """
    zero_string = "0" * num_qubits
    return counts.get(zero_string, 0) / shots


def build_measurement_circuit(bound_circuit: QuantumCircuit) -> QuantumCircuit:
    circ = bound_circuit.copy()
    circ.measure_all()
    return circ


def build_compute_uncompute_circuit(
    feature_map: QuantumCircuit, x_params, xi: np.ndarray, xj: np.ndarray
) -> QuantumCircuit:
    """Build the standard compute-uncompute fidelity test circuit: apply
    U(xi), then the inverse of U(xj), to the all-zero state. The
    probability of measuring all zeros afterward equals the fidelity
    between the two encoded states.
    """
    n = feature_map.num_qubits
    circ = QuantumCircuit(n)
    circ.compose(feature_map.assign_parameters(dict(zip(x_params, xi))), inplace=True)
    circ.compose(feature_map.assign_parameters(dict(zip(x_params, xj))).inverse(), inplace=True)
    circ.measure_all()
    return circ
