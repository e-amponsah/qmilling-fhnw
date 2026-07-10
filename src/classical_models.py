"""Leakage-free classical baselines, run through the shared LOOCV harness.

Each factory returns a fresh, unfitted sklearn estimator; `evaluation.py`
is responsible for fitting/scaling strictly within each LOOCV fold.
"""

import logging

from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC

from src.config import RANDOM_SEED
from src.evaluation import loocv_evaluate, loocv_evaluate_regression, summarize_results

logger = logging.getLogger(__name__)

# --- Model factories -------------------------------------------------------

def make_svc() -> CalibratedClassifierCV:
    # SVC(probability=True)'s internal Platt scaling is deprecated as of
    # sklearn 1.9; CalibratedClassifierCV is the maintainer-recommended
    # replacement for a probability-calibrated SVC.
    base = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_SEED)
    return CalibratedClassifierCV(base, method="sigmoid", cv=3, ensemble=False)


def make_random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=200, max_depth=4, random_state=RANDOM_SEED)


def make_gradient_boosting() -> GradientBoostingClassifier:
    return GradientBoostingClassifier(n_estimators=100, max_depth=2, random_state=RANDOM_SEED)


def make_pls(n_components: int = 2) -> PLSRegression:
    return PLSRegression(n_components=n_components)


CLASSICAL_CLASSIFIERS = {
    "SVC_rbf": make_svc,
    "RandomForest": make_random_forest,
    "GradientBoosting": make_gradient_boosting,
}


def run_classical_classification_suite(X, y) -> dict:
    """Run every registered classical classifier through LOOCV; returns
    {model_name: loocv_evaluate(...) output}.
    """
    results = {}
    for name, factory in CLASSICAL_CLASSIFIERS.items():
        logger.info("Running classical classifier: %s", name)
        results[name] = loocv_evaluate(factory, X, y)
        logger.info("  %s -> accuracy=%.3f f1=%.3f",
                     name, results[name]["metrics"]["accuracy"], results[name]["metrics"]["f1"])
    return results


def run_pls_regression_baseline(X, y_reg, n_components: int = 2) -> dict:
    """PLS regression baseline for direct comparison against the Patzmann
    et al. Q^2=0.77 benchmark (regression framing of the same problem).
    """
    logger.info("Running PLS regression baseline (n_components=%d)", n_components)
    result = loocv_evaluate_regression(lambda: make_pls(n_components), X, y_reg)
    logger.info("  PLS -> Q2(LOOCV)=%.3f R2(train)=%.3f",
                 result["metrics"]["q2_loocv"], result["metrics"]["r2_train"])
    return result


if __name__ == "__main__":
    import pandas as pd

    from src.config import CLASSIFICATION_TARGET, MODELING_TABLE_CSV, REGRESSION_TARGET
    from src.features import get_candidate_matrix, select_k_best_features

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    data = pd.read_csv(MODELING_TABLE_CSV)
    X_full = get_candidate_matrix(data)
    y_clf = data[CLASSIFICATION_TARGET]
    y_reg = data[REGRESSION_TARGET]

    selection = select_k_best_features(X_full, y_clf)
    X = X_full[selection["selected_features"]]
    logger.info("Selected features: %s", selection["selected_features"])

    clf_results = run_classical_classification_suite(X, y_clf)
    table = summarize_results(clf_results, kind="classification")
    print(table)

    run_pls_regression_baseline(X, y_reg)
