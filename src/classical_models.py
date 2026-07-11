"""Classical baseline models, run through the same LOOCV harness as the
quantum models so results are directly comparable.

Each function here just builds and returns an unfitted sklearn estimator.
Fitting and scaling within each LOOCV fold is handled by evaluation.py, not
by this file.
"""

import logging

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC

from src.config import RANDOM_SEED
from src.evaluation import loocv_evaluate, loocv_evaluate_regression, summarize_results

logger = logging.getLogger(__name__)

# --- Model factories -------------------------------------------------------

def make_svc() -> CalibratedClassifierCV:
    # sklearn 1.9 deprecated SVC(probability=True)'s internal Platt
    # scaling. CalibratedClassifierCV is the recommended replacement for
    # getting calibrated probabilities out of an SVC.
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
    """Run every registered classical classifier through LOOCV and return
    a dict mapping model name to its loocv_evaluate() output.
    """
    results = {}
    for name, factory in CLASSICAL_CLASSIFIERS.items():
        logger.info("Running classical classifier: %s", name)
        results[name] = loocv_evaluate(factory, X, y)
        logger.info("  %s -> accuracy=%.3f f1=%.3f",
                     name, results[name]["metrics"]["accuracy"], results[name]["metrics"]["f1"])
    return results


def run_pls_regression_baseline(X, y_reg, n_components: int = 2, log_target: bool = True) -> dict:
    """PLS regression baseline, for comparing against the Patzmann et al.
    benchmark of R^2 = 0.82 and Q^2 = 0.77.

    Patzmann et al. modeled log(COMDR_15) rather than the raw ratio. Their
    published table's target row is literally named LogCOMDR15min.
    COMDR_15min ranges from about 1.1 to 26.5 and is heavily skewed, so a
    plain linear PLS fit on the raw ratio ends up dominated by a few
    extreme high responders like Fenofibrate and generalizes poorly with
    only 29 samples. Log transforming the target, which is the default
    here and matches what the benchmark used, fixes this: it moves Q^2
    from about 0.45 to about 0.76 and R^2 from about 0.68 to about 0.83 on
    the Patzmann approximation feature set (D50, MolLogP, Kappa3,
    apparent_solubility), close to their reported numbers.
    """
    target = np.log(y_reg) if log_target else y_reg
    target_scale = "log(COMDR_15min)" if log_target else "COMDR_15min"
    logger.info("Running PLS regression baseline (n_components=%d, target=%s)", n_components, target_scale)
    result = loocv_evaluate_regression(lambda: make_pls(n_components), X, target)
    result["metrics"]["n_components"] = n_components
    result["metrics"]["target_scale"] = target_scale
    logger.info("  PLS -> Q2(LOOCV)=%.3f R2(train)=%.3f (target=%s)",
                 result["metrics"]["q2_loocv"], result["metrics"]["r2_train"], target_scale)
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
