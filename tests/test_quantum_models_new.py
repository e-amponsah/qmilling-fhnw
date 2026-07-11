"""Smoke tests for the 5 new quantum models added on top of the core suite:
QK-KRR, VQR, QCNN-R (regression) and QEC, QCHB (classification).

These run against a local, noiseless AerSimulator with a small shot count
and tiny synthetic data (a handful of samples), so they finish in seconds --
they check that each model's `fit`/`predict` pipeline runs end to end and
produces outputs of the right shape/type, not that it reaches any
particular accuracy. Real-quality LOOCV numbers come from `main.py
--stage quantum` against the actual 29-drug dataset.
"""

import numpy as np
import pytest

from src.quantum_backend import ExecutionConfig, QuantumExecutor
from src.quantum_models import (
    QCNNRegressor,
    QuantumClassicalHybridBoosting,
    QuantumEnsembleClassifier,
    QuantumKernelRidgeRegression,
    VariationalQuantumRegressor,
)

N_SAMPLES = 6


@pytest.fixture(scope="module")
def executor_aer() -> QuantumExecutor:
    """One shared noiseless-simulator executor for every test in this
    module, mirroring how a real LOOCV run shares one executor across folds.
    Low shot count keeps these tests fast; accuracy is not being checked.
    """
    return QuantumExecutor(ExecutionConfig(mode="aer_simulator", shots=256))


@pytest.fixture
def synthetic_4feature():
    rng = np.random.RandomState(0)
    X = rng.uniform(0, np.pi, size=(N_SAMPLES, 4))
    y_clf = np.array([0, 1, 0, 1, 0, 1])
    y_reg = rng.uniform(1.0, 10.0, size=N_SAMPLES)
    return X, y_clf, y_reg


@pytest.fixture
def synthetic_6feature():
    rng = np.random.RandomState(1)
    X = rng.uniform(0, np.pi, size=(N_SAMPLES, 6))
    y_reg = rng.uniform(1.0, 10.0, size=N_SAMPLES)
    return X, y_reg


def test_qk_krr_smoke(executor_aer, synthetic_4feature):
    """QK-KRR trains and predicts on synthetic samples without error, and
    reports its expressibility diagnostics.
    """
    X, _, y_reg = synthetic_4feature
    model = QuantumKernelRidgeRegression(
        executor=executor_aer, feature_map_name="zzplus", alpha=0.1, alpha_grid=(0.1,)
    )
    model.fit(X[:5], y_reg[:5])
    pred = model.predict(X[5:])

    assert pred.shape == (1,)
    assert np.all(np.isfinite(pred))
    assert model.kernel_frobenius_norm_ > 0
    assert model.kernel_effective_rank_ > 0


def test_vqr_output_range(executor_aer, synthetic_4feature):
    """VQR predicts a finite, real value in a reasonable range for
    COMDR15 (which spans roughly [0, 15] in the actual dataset) given
    training targets in that same neighborhood.
    """
    X, _, y_reg = synthetic_4feature
    model = VariationalQuantumRegressor(
        executor=executor_aer, n_layers=1, n_output_qubits=2, maxiter_spsa=2, maxiter_cobyla=2
    )
    model.fit(X[:5], y_reg[:5])
    pred = model.predict(X[5:])

    assert pred.shape == (1,)
    assert np.all(np.isfinite(pred))
    # A Ridge head fit on training targets in [1, 10] should not blow up to
    # wildly implausible values on a single nearby test point.
    assert -20.0 < pred[0] < 30.0


def test_qcnn_regressor_smoke(executor_aer, synthetic_6feature):
    """QCNN-R (5th model) trains and predicts on its required 6-feature
    input, and rejects a mismatched feature count.
    """
    X, y_reg = synthetic_6feature
    model = QCNNRegressor(executor=executor_aer, n_qubits=6, maxiter_spsa=2, maxiter_cobyla=2)
    model.fit(X[:5], y_reg[:5])
    pred = model.predict(X[5:])

    assert pred.shape == (1,)
    assert np.all(np.isfinite(pred))

    with pytest.raises(ValueError):
        QCNNRegressor(executor=executor_aer, n_qubits=6).fit(X[:5, :4], y_reg[:5])


def test_qec_weights_sum_to_one(executor_aer, synthetic_4feature):
    """The ensemble's per-member weights sum to 1 and each lies in (0, 1)."""
    X, y_clf, _ = synthetic_4feature
    model = QuantumEnsembleClassifier(executor=executor_aer, C=1.0, kta_temperature=0.5)
    model.fit(X[:5], y_clf[:5])

    assert model._weights is not None
    assert abs(model._weights.sum() - 1.0) < 1e-9
    assert all(0.0 < w < 1.0 for w in model._weights)
    assert len(model.ktas_) == 3

    proba = model.predict_proba(X[5:])
    assert proba.shape == (1, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_qchb_alphas_positive(executor_aer, synthetic_4feature):
    """Every round's learner weight alpha_t is strictly positive, i.e. every
    round contributes a valid (non-inverted) vote to the boosted ensemble.
    """
    X, y_clf, _ = synthetic_4feature
    model = QuantumClassicalHybridBoosting(executor=executor_aer, n_rounds=2, n_layers=1, maxiter=8)
    model.fit(X[:5], y_clf[:5])

    assert len(model.alphas_) == 2
    assert all(alpha > 0 for alpha in model.alphas_)
    assert all(0.0 <= eps <= 1.0 for eps in model.round_errors_)

    pred = model.predict(X[5:])
    assert pred.shape == (1,)
    assert pred[0] in (0, 1)
