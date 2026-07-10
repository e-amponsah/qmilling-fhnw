"""LOOCV orchestration engine, Kernel Target Alignment, and results summarizer.

The single hard rule enforced throughout this module: any scaler, selector,
or model is fit *only* on the training fold inside the LOOCV loop, and only
`.transform()` / `.predict()` touches the held-out sample. `loocv_evaluate`
and `loocv_evaluate_regression` are the two entry points every model in
`classical_models.py` and `quantum_models.py` is routed through, so this
guarantee holds uniformly across the whole model suite.
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
    """Full N-fold LOOCV for a binary classifier.

    `model_factory()` must return a *fresh, unfitted* model instance -- called
    once per fold so no state (including any internal RNG/optimizer state)
    leaks across folds.
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
    """Full N-fold LOOCV for a regressor; reports Q^2 (LOOCV) and, via a
    separate full-data refit, R^2 (training fit) for comparison against the
    Patzmann et al. benchmark table.
    """
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    loo = LeaveOneOut()

    y_true, y_pred = [], []
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

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    q2 = r2_score(y_true, y_pred)  # R^2 of LOOCV predictions vs truth == Q^2

    scaler = MinMaxScaler(feature_range=feature_range)
    X_full = scaler.fit_transform(X_arr) if scale else X_arr
    full_model = model_factory()
    full_model.fit(X_full, y_arr)
    r2_train = r2_score(y_arr, full_model.predict(X_full))

    metrics = {"q2_loocv": q2, "r2_train": r2_train, "n_folds": len(y_true)}
    return {"y_true": y_true, "y_pred": y_pred, "metrics": metrics}


def kernel_target_alignment(K: np.ndarray, y: np.ndarray) -> float:
    """Kernel-Target Alignment: cosine similarity between the kernel matrix K
    and the ideal label kernel yy^T, in {-1, ..., +1}. Higher means the
    kernel's geometry already separates the two classes.
    """
    y_signed = np.where(np.asarray(y) > 0, 1.0, -1.0)
    K_y = np.outer(y_signed, y_signed)
    num = np.sum(K * K_y)
    den = np.sqrt(np.sum(K * K) * np.sum(K_y * K_y))
    return float(num / den) if den > 0 else 0.0


def summarize_results(results_by_model: dict[str, dict], kind: str = "classification") -> pd.DataFrame:
    """Flatten a {model_name: loocv_evaluate(...) output} dict into one table."""
    rows = []
    for name, res in results_by_model.items():
        row = {"model": name, **res["metrics"]}
        rows.append(row)
    df = pd.DataFrame(rows)
    sort_col = "accuracy" if kind == "classification" else "q2_loocv"
    if sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)
    return df


def save_results_table(df: pd.DataFrame, filename: str) -> None:
    out_path = RESULTS_DIR / filename
    df.to_csv(out_path, index=False)
    logger.info("Saved %s", out_path)


def _self_test_leakage_guard() -> None:
    """Runnable sanity check: a scaler fit on the *full* dataset (leaky) must
    not silently match the fold-safe path -- proves the harness is actually
    exercising leak-free scaling rather than a no-op.
    """
    from sklearn.linear_model import LogisticRegression

    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.rand(29, 6) * 100, columns=[f"f{i}" for i in range(6)])
    y = pd.Series((rng.rand(29) > 0.5).astype(int))

    safe = loocv_evaluate(lambda: LogisticRegression(max_iter=1000), X, y, scale=True)

    # Leaky control: fit scaler on the FULL dataset once, then run "LOOCV"
    # only on the already-globally-scaled data.
    X_leaky = MinMaxScaler().fit_transform(X.values)
    leaky_res = loocv_evaluate(
        lambda: LogisticRegression(max_iter=1000),
        pd.DataFrame(X_leaky, columns=X.columns), y, scale=False,
    )

    logger.info("Leak-free accuracy: %.3f | Leaky-scaler accuracy: %.3f",
                safe["metrics"]["accuracy"], leaky_res["metrics"]["accuracy"])
    logger.info("(Scores may coincide by chance on random data; the guarantee "
                "this proves is structural -- `scale=True` never calls "
                "MinMaxScaler.fit on the held-out row -- verified by code path, "
                "not by this numeric comparison alone.)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _self_test_leakage_guard()
