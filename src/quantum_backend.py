"""Real circuit execution layer: every quantum computation in this project
(kernel fidelities, variational-model expectation values) is submitted as an
actual `SamplerV2` job -- transpiled to a backend's ISA and run with finite
shots -- never a shortcut linear-algebra simulation. The only thing that
changes between "local" and "on IBM hardware" is which `Backend` object the
Sampler is pointed at.

Toggle via `ExecutionConfig.mode`:
    "aer_simulator" : local, noiseless AerSimulator (default -- fast, free).
    "aer_noisy"     : local AerSimulator loaded with a device-like noise model.
    "ibm_runtime"   : a real (or cloud-simulated) backend via
                       `QiskitRuntimeService`, using the account saved by
                       `scripts/setup_ibm_account.py`.

Every quantum model class in `quantum_models.py` takes an `ExecutionConfig`
and is otherwise agnostic to where its circuits actually run.
"""

import logging
import os
from dataclasses import dataclass, field

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
    """A synthetic but realistic near-term-hardware noise model: depolarizing
    error on single-/two-qubit gates plus symmetric readout error, in the
    ballpark of current superconducting-qubit device calibration reports.
    Used for `mode="aer_noisy"` when no live IBM backend is configured.
    """
    noise_model = NoiseModel(basis_gates=["u", "cx"])
    noise_model.add_all_qubit_quantum_error(depolarizing_error(single_qubit_error, 1), ["u"])
    noise_model.add_all_qubit_quantum_error(depolarizing_error(two_qubit_error, 2), ["cx"])
    ro_error = ReadoutError([[1 - readout_error, readout_error], [readout_error, 1 - readout_error]])
    noise_model.add_all_qubit_readout_error(ro_error)
    return noise_model


@dataclass
class ExecutionConfig:
    """Everything needed to route a batch of circuits to an actual backend."""

    mode: str = "aer_simulator"  # "aer_simulator" | "aer_noisy" | "ibm_runtime"
    shots: int = 4096
    optimization_level: int = 1
    seed: int = RANDOM_SEED
    ibm_backend_name: str | None = None  # None -> least-busy operational backend
    noise_model: NoiseModel | None = None  # override for "aer_noisy"; else built lazily

    def label(self) -> str:
        if self.mode == "ibm_runtime":
            return f"ibm_runtime[{self.ibm_backend_name or 'least_busy'}]"
        return self.mode


def default_execution_config() -> ExecutionConfig:
    """Reads QC_BACKEND_MODE / QC_SHOTS / QC_IBM_BACKEND from the environment
    so the execution target can be toggled without touching code (e.g. a
    `.env` file loaded by `scripts/setup_ibm_account.py`, or an inline
    `QC_BACKEND_MODE=ibm_runtime python main.py ...`).
    """
    mode = os.environ.get("QC_BACKEND_MODE", "aer_simulator")
    shots = int(os.environ.get("QC_SHOTS", "4096"))
    ibm_backend_name = os.environ.get("QC_IBM_BACKEND") or None
    return ExecutionConfig(mode=mode, shots=shots, ibm_backend_name=ibm_backend_name)


class QuantumExecutor:
    """Resolves an `ExecutionConfig` to a live backend + Sampler once, then
    runs batches of circuits against it. One instance should be reused across
    an entire LOOCV run (or at least a full fold) to avoid re-resolving the
    IBM Runtime service / re-picking a least-busy backend on every call.
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
        """Transpile + submit every circuit as ONE Sampler job, return a list
        of measurement-count dicts (bitstring -> count) in input order.
        Batching every circuit needed for a LOOCV fold (or a full kernel
        matrix) into a single job is what makes real hardware/cloud queue
        execution remotely practical instead of one-job-per-circuit.
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
    """P(qubit == 1) from a Sampler counts dict. Qiskit bitstrings are
    little-endian (rightmost character = qubit 0).
    """
    ones = sum(c for bitstring, c in counts.items() if bitstring[::-1][qubit] == "1")
    return ones / shots


def fidelity_from_counts(counts: dict[str, int], num_qubits: int, shots: int) -> float:
    """Compute-uncompute fidelity estimate: P(measuring the all-zeros string)."""
    zero_string = "0" * num_qubits
    return counts.get(zero_string, 0) / shots


def build_measurement_circuit(bound_circuit: QuantumCircuit) -> QuantumCircuit:
    circ = bound_circuit.copy()
    circ.measure_all()
    return circ


def build_compute_uncompute_circuit(
    feature_map: QuantumCircuit, x_params, xi: np.ndarray, xj: np.ndarray
) -> QuantumCircuit:
    """U(xi) then U(xj)^-1 applied to |0>; P(all-zeros) on measurement = fidelity."""
    n = feature_map.num_qubits
    circ = QuantumCircuit(n)
    circ.compose(feature_map.assign_parameters(dict(zip(x_params, xi))), inplace=True)
    circ.compose(feature_map.assign_parameters(dict(zip(x_params, xj))).inverse(), inplace=True)
    circ.measure_all()
    return circ
