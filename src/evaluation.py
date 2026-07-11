"""The LOOCV evaluation engine, the Kernel Target Alignment metric, and the
results table builder.

Any scaler or model is fit only on the training fold, and the held out 
sample only ever gets .transform() or .predict() called on it, never .fit(). 
loocv_evaluate and loocv_evaluate_regression are the two functions that every 
model in classical_models.py and quantum_models.py runs through, so this rule holds
for the whole project, not just some models.
"""

import logging
import time
from typing import Callable, Protocol

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, r2_score, recall_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import MinMaxScaler

from src.config import RESULTS_DIR

logger = logging.getLogger(__name__)


class Classifier(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> "Classifier": ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


def loocv_evaluate(
    model_factory: Callable[[], Classifier],
    X: pd.DataFrame,
    y: pd.Series,
    feature_range: tuple[float, float] = (0.0, 1.0),
    scale: bool = True,
) -> dict:
    """Run full leave one out cross validation for a binary classifier.

    model_factory() must return a fresh, unfitted model each time it is
    called, since it gets called once per fold. This makes sure no state
    from one fold, including any random seed or optimizer state, carries
    over into the next fold.
    """
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y)
    loo = LeaveOneOut()

    y_true, y_pred, y_proba = [], [], []
    t0 = time.perf_counter()
    for train_idx, test_idx in loo.split(X_arr):
        X_train, X_test = X_arr[train_idx], X_arr[test_idx]
        y_train, y_test = y_arr[train_idx], y_arr[test_idx]

        if scale:
            scaler = MinMaxScaler(feature_range=feature_range)
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        model = model_factory()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        y_pred.append(int(pred[0]))
        y_true.append(int(y_test[0]))

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)
            y_proba.append(float(np.asarray(proba).reshape(-1)[-1]))
        else:
            y_proba.append(float(pred[0]))

    elapsed = time.perf_counter() - t0
    y_true, y_pred, y_proba = np.array(y_true), np.array(y_pred), np.array(y_proba)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "recall_class0": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "recall_class1": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "n_folds": len(y_true),
        "elapsed_s": elapsed,
    }
    return {"y_true": y_true, "y_pred": y_pred, "y_proba": y_proba, "metrics": metrics}


def loocv_evaluate_regression(
    model_factory: Callable[[], object],
    X: pd.DataFrame,
    y: pd.Series,
    feature_range: tuple[float, float] = (0.0, 1.0),
    scale: bool = True,
) -> dict:
    """Run full leave one out cross validation for a regressor. Reports Q^2
    from the LOOCV predictions, and separately reports R^2 from refitting
    on the full dataset, so both numbers can be compared to the Patzmann
    et al. benchmark table.
    """
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    loo = LeaveOneOut()

    y_true, y_pred = [], []
    t0 = time.perf_counter()
    for train_idx, test_idx in loo.split(X_arr):
        X_train, X_test = X_arr[train_idx], X_arr[test_idx]
        y_train, y_test = y_arr[train_idx], y_arr[test_idx]

        if scale:
            scaler = MinMaxScaler(feature_range=feature_range)
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        model = model_factory()
        model.fit(X_train, y_train)
        y_pred.append(float(np.asarray(model.predict(X_test)).reshape(-1)[0]))
        y_true.append(float(y_test[0]))
    elapsed = time.perf_counter() - t0

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    q2 = r2_score(y_true, y_pred)  # Q^2 is just R^2 computed on the LOOCV predictions

    scaler = MinMaxScaler(feature_range=feature_range)
    X_full = scaler.fit_transform(X_arr) if scale else X_arr
    full_model = model_factory()
    full_model.fit(X_full, y_arr)
    r2_train = r2_score(y_arr, full_model.predict(X_full))

    metrics = {"q2_loocv": q2, "r2_train": r2_train, "n_folds": len(y_true), "elapsed_s": elapsed}
    return {"y_true": y_true, "y_pred": y_pred, "metrics": metrics}


def kernel_target_alignment(K: np.ndarray, y: np.ndarray) -> float:
    """Kernel Target Alignment: the cosine similarity between the kernel
    matrix K and the ideal label kernel y times y transpose. Ranges from
    -1 to 1. A higher value means the kernel's geometry already lines up
    with the class labels.
    """
    y_signed = np.where(np.asarray(y) > 0, 1.0, -1.0)
    K_y = np.outer(y_signed, y_signed)
    num = np.sum(K * K_y)
    den = np.sqrt(np.sum(K * K) * np.sum(K_y * K_y))
    return float(num / den) if den > 0 else 0.0


def continuous_kernel_target_alignment(K: np.ndarray, y: np.ndarray) -> float:
    """Continuous-target counterpart of `kernel_target_alignment`, for
    regression rather than classification (used by
    `TrainedQuantumKernelRidgeRegression`).

    The classification version compares K to the ideal kernel
    y_signed @ y_signed.T, where y_signed is +-1. There is no natural
    +-1 label for a continuous target, so this instead uses the ideal
    kernel Y = y_c @ y_c.T, where y_c = y - mean(y) is the centered
    target. Y[i, j] = y_c[i] * y_c[j] is positive and large when i and j
    deviate from the target's mean in the SAME direction (both high or
    both low responders), negative when they deviate in OPPOSITE
    directions, and near zero whenever either sits close to the mean.
    Maximizing alignment with Y therefore pushes a kernel toward giving
    high fidelity to pairs that deviate similarly and low fidelity to
    pairs that deviate oppositely -- the geometry a downstream
    KernelRidge head needs in order to interpolate y from its neighbours.

    K is double-centered (Cortes et al., 2012) before comparison, removing
    any constant kernel-wide offset (e.g. a fidelity kernel's
    diagonal-heavy bias) that would otherwise dominate a raw, uncentered
    alignment. Y needs no separate centering: since y_c already has zero
    mean, every row/column mean of y_c @ y_c.T is already exactly zero, so
    double-centering it would be a no-op.
    """
    y_c = np.asarray(y, dtype=float) - np.mean(y)
    Y = np.outer(y_c, y_c)

    n = K.shape[0]
    ones_over_n = np.full((n, n), 1.0 / n)
    K_c = K - ones_over_n @ K - K @ ones_over_n + ones_over_n @ K @ ones_over_n

    num = np.sum(K_c * Y)
    den = np.sqrt(np.sum(K_c * K_c) * np.sum(Y * Y))
    return float(num / den) if den > 0 else 0.0


def summarize_results(results_by_model: dict[str, dict], kind: str = "classification") -> pd.DataFrame:
    """Turn a dict of model name to loocv_evaluate() output into one table."""
    rows = []
    for name, res in results_by_model.items():
        row = {"model": name, **res["metrics"]}
        rows.append(row)
    df = pd.DataFrame(rows)
    if kind == "classification":
        sort_col = "accuracy"
    else:
        # "q2_loocv" for loocv_evaluate_regression() output (e.g. PLS),
        # "q2" for run_quantum_regression_suite() output -- both name the
        # same LOOCV-R^2 quantity, just under the key each caller settled on.
        sort_col = "q2_loocv" if "q2_loocv" in df.columns else "q2"
    if sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)
    return df


def save_results_table(df: pd.DataFrame, filename: str) -> None:
    out_path = RESULTS_DIR / filename
    df.to_csv(out_path, index=False)
    logger.info("Saved %s", out_path)


def _self_test_leakage_guard() -> None:
    """A quick check comparing the leak free path against a leaky control,
    where the scaler is fit on the whole dataset before LOOCV even starts.
    On random data the two accuracy numbers may end up close by chance.
    What actually matters is that scale=True never calls scaler.fit() on
    the held out row, which is guaranteed by the code itself, not by this
    comparison. This check is here as a runnable sanity test, not as proof.
    """
    from sklearn.linear_model import LogisticRegression

    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.rand(29, 6) * 100, columns=[f"f{i}" for i in range(6)])
    y = pd.Series((rng.rand(29) > 0.5).astype(int))

    safe = loocv_evaluate(lambda: LogisticRegression(max_iter=1000), X, y, scale=True)

    # Leaky control: fit the scaler on the whole dataset once, up front,
    # then run "LOOCV" on data that has already seen every row.
    X_leaky = MinMaxScaler().fit_transform(X.values)
    leaky_res = loocv_evaluate(
        lambda: LogisticRegression(max_iter=1000),
        pd.DataFrame(X_leaky, columns=X.columns), y, scale=False,
    )

    logger.info("Leak-free accuracy: %.3f, leaky-scaler accuracy: %.3f",
                safe["metrics"]["accuracy"], leaky_res["metrics"]["accuracy"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _self_test_leakage_guard()
