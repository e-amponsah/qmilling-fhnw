"""Builds molecular descriptors, assembles the modeling table, and selects
features for training. Feature selection is done fold by fold so that no
information from the held-out sample ever influences which columns are
picked.
"""

import logging

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.GraphDescriptors import BertzCT
from rdkit.Chem.rdMolDescriptors import CalcChi0v, CalcChi1v, CalcKappa1, CalcKappa2, CalcKappa3
from sklearn.feature_selection import VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import MinMaxScaler

from src.config import (
    CANDIDATE_FEATURE_COLS,
    CLASSIFICATION_TARGET,
    DATASET_CSV,
    DESCRIPTOR_COLS,
    DESCRIPTORS_CSV,
    FEATURE_SELECT_K_RANGE,
    LEAKAGE_COLS,
    MODELING_TABLE_CSV,
    PATZMANN_APPROX_COLS,
    PATZMANN_MISSING_COLS,
    RANDOM_SEED,
    REGRESSION_TARGET,
)
from src.pca_analysis import get_pc_dominant_cluster, redundancy_aware_topk, run_pca

logger = logging.getLogger(__name__)


def compute_descriptors(smiles: str) -> dict:
    """Return the 14 RDKit descriptors for one SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    return {
        "MolLogP": Descriptors.MolLogP(mol),
        "MolWt": Descriptors.MolWt(mol),
        "TPSA": Descriptors.TPSA(mol),
        "NumHAcceptors": rdMolDescriptors.CalcNumHBA(mol),
        "NumHDonors": rdMolDescriptors.CalcNumHBD(mol),
        "NumRotatableBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "Kappa1": CalcKappa1(mol),
        "Kappa2": CalcKappa2(mol),
        "Kappa3": CalcKappa3(mol),
        "Chi0v": CalcChi0v(mol),
        "Chi1v": CalcChi1v(mol),
        "BertzCT": BertzCT(mol),
        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
        "RingCount": rdMolDescriptors.CalcNumRings(mol),
    }


def build_descriptor_table(smiles_df: pd.DataFrame, out_path=DESCRIPTORS_CSV) -> pd.DataFrame:
    """Compute the 14 descriptors for every drug in `smiles_df` (cols: drug, SMILES)."""
    rows = []
    for _, row in smiles_df.iterrows():
        rows.append({"drug": row["drug"], **compute_descriptors(row["SMILES"])})

    descriptors = pd.DataFrame(rows)[["drug"] + DESCRIPTOR_COLS].round(4)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    descriptors.to_csv(out_path, index=False)
    logger.info("Saved %s (%d rows x %d cols)", out_path, *descriptors.shape)
    return descriptors


def build_modeling_table(
    descriptors: pd.DataFrame,
    dataset_path=DATASET_CSV,
    out_path=MODELING_TABLE_CSV,
) -> pd.DataFrame:
    """Merge the RDKit descriptors with the experimental dataset.

    COM_15min and PM_15min are the raw measurements that COMDR_15min is
    computed from, so they stay in this merged table but must never be
    passed into a model as a feature. That filtering happens in
    get_candidate_matrix below, not here.
    """
    dataset = pd.read_csv(dataset_path)
    data = descriptors.merge(dataset, on="drug", how="inner")

    if len(data) != len(dataset):
        missing = set(dataset["drug"]) - set(descriptors["drug"])
        raise ValueError(f"Merge dropped rows; missing descriptors for: {missing}")

    missing_from_table = [c for c in PATZMANN_APPROX_COLS if c not in data.columns]
    if missing_from_table:
        raise ValueError(f"Expected Patzmann-approximation columns missing: {missing_from_table}")
    if PATZMANN_MISSING_COLS:
        logger.info(
            "Note: Patzmann et al. reference variable(s) %s are not present in the "
            "provided dataset. Only %s are reproducible here.",
            PATZMANN_MISSING_COLS, PATZMANN_APPROX_COLS,
        )
    else:
        logger.info(
            "All four Patzmann et al. reference variables are reproducible from the "
            "provided data: %s.", PATZMANN_APPROX_COLS,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(out_path, index=False)
    logger.info("Saved %s (%d rows x %d cols)", out_path, *data.shape)
    return data


def get_candidate_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """Return the candidate feature matrix: the 14 RDKit descriptors plus
    D50 and apparent_solubility. Raises if a leakage column has somehow
    ended up in the candidate list, as a safety check.
    """
    leaked = [c for c in LEAKAGE_COLS if c in CANDIDATE_FEATURE_COLS]
    if leaked:
        raise RuntimeError(f"Leakage columns present in candidate feature pool: {leaked}")
    return data[CANDIDATE_FEATURE_COLS].copy()


def _pca_cluster_topk(X_arr: np.ndarray, y_arr: np.ndarray, feature_names: np.ndarray, k: int, n_top_pcs: int = 3) -> list[str]:
    """Score features with a univariate f_classif test, then pick the top k
    using redundancy_aware_topk so that features loading on the same PCA
    cluster do not all get picked together. The caller must pass in
    training fold data only, since this fits a PCA model internally.
    """
    k_eff = min(k, X_arr.shape[1])
    f_scores, _ = f_classif(X_arr, y_arr)
    f_scores = np.nan_to_num(f_scores, nan=0.0)

    _, _, loadings = run_pca(pd.DataFrame(X_arr, columns=feature_names))
    clusters = get_pc_dominant_cluster(loadings, n_top_pcs=n_top_pcs)

    return redundancy_aware_topk(f_scores, feature_names, clusters, k_eff)


def select_k_best_features(
    X: pd.DataFrame,
    y: pd.Series,
    k_range: tuple[int, int] = FEATURE_SELECT_K_RANGE,
    variance_threshold: float = 1e-6,
    random_state: int = RANDOM_SEED,
) -> dict:
    """Automated feature selection, done separately inside each LOOCV fold.

    For every candidate k in k_range, this runs a full nested LOOCV: for
    each training fold, a VarianceThreshold filter, an f_classif score, and
    a PCA fit (see pca_analysis.py) are all computed on that fold's training
    data only, then used to score a simple classifier on the held-out
    sample. This keeps both the feature selection and the PCA redundancy
    check from ever seeing the held-out sample.

    Feature selection prefers one feature per PCA cluster before picking a
    second feature from any cluster. For example, MolWt, Chi0v, Chi1v,
    Kappa1, Kappa2, Kappa3, and BertzCT all measure roughly the same thing
    (molecule size), so taking four of them barely adds more information
    than taking one of them and using the other three slots on unrelated
    descriptors.

    Returns a dict with the best k, the LOOCV score for every k, and the
    final feature list chosen by running selection once on the full
    dataset. That final list is for reporting and for models with a fixed
    input size (like the 6 qubit QCNN). Any actual model evaluation must
    redo selection per fold using the nested loop above, not this final list.
    """
    X_arr = X.values
    y_arr = y.values
    feature_names = np.array(X.columns)
    loo = LeaveOneOut()

    scores_by_k = {}
    for k in range(k_range[0], k_range[1] + 1):
        fold_correct = 0
        for train_idx, test_idx in loo.split(X_arr):
            X_train, X_test = X_arr[train_idx], X_arr[test_idx]
            y_train, y_test = y_arr[train_idx], y_arr[test_idx]

            vt = VarianceThreshold(threshold=variance_threshold)
            X_train_vt = vt.fit_transform(X_train)
            X_test_vt = vt.transform(X_test)
            kept_names_fold = feature_names[vt.get_support()]

            selected = _pca_cluster_topk(X_train_vt, y_train, kept_names_fold, k)
            col_idx = [list(kept_names_fold).index(f) for f in selected]
            X_train_sel = X_train_vt[:, col_idx]
            X_test_sel = X_test_vt[:, col_idx]

            scaler = MinMaxScaler(feature_range=(0.0, 1.0))
            X_train_scaled = scaler.fit_transform(X_train_sel)
            X_test_scaled = scaler.transform(X_test_sel)

            clf = LogisticRegression(max_iter=1000, random_state=random_state)
            clf.fit(X_train_scaled, y_train)
            pred = clf.predict(X_test_scaled)
            fold_correct += int(pred[0] == y_test[0])

        scores_by_k[k] = fold_correct / len(X_arr)

    best_k = max(scores_by_k, key=scores_by_k.get)

    # Fit selection once more on the full dataset, just to report which
    # feature names were chosen and to give the QCNN a fixed 6 feature
    # list. This is not used for scoring any model.
    vt = VarianceThreshold(threshold=variance_threshold)
    X_vt = vt.fit_transform(X_arr)
    kept_names = feature_names[vt.get_support()]

    features_by_k = {}
    for k in range(k_range[0], k_range[1] + 1):
        features_by_k[k] = _pca_cluster_topk(X_vt, y_arr, kept_names, k)

    return {
        "best_k": best_k,
        "scores_by_k": scores_by_k,
        "selected_features": features_by_k[best_k],
        "features_by_k": features_by_k,
    }


def _resolve_qcnn_features(X: pd.DataFrame, y: pd.Series, manual_features_qcnn: list[str] | None) -> list[str]:
    """Resolve the fixed 6-feature subset used by the QCNN / QCNN-R models.

    If MANUAL_FEATURES_QCNN is set in .env, it is validated (must be exactly
    6 known columns) and used directly. Otherwise falls back to an automated
    6-feature selection, independent of whichever feature count the other
    models ended up using.
    """
    if manual_features_qcnn:
        if len(manual_features_qcnn) != 6:
            raise ValueError(
                f"MANUAL_FEATURES_QCNN (.env) has {len(manual_features_qcnn)} feature(s), but the "
                f"QCNN needs exactly 6 (one per qubit): {manual_features_qcnn}"
            )
        invalid = [f for f in manual_features_qcnn if f not in X.columns]
        if invalid:
            raise ValueError(
                f"MANUAL_FEATURES_QCNN (.env) contains unknown column(s): {invalid}. "
                f"Valid candidates are: {list(X.columns)}"
            )
        logger.info("Using manual QCNN feature override from .env: %s", manual_features_qcnn)
        return list(manual_features_qcnn)
    return select_k_best_features(X, y, k_range=(6, 6))["features_by_k"][6]


def get_modeling_features(
    X: pd.DataFrame, y: pd.Series, k_range: tuple[int, int] = FEATURE_SELECT_K_RANGE, **kwargs
) -> dict:
    """Main feature selection entry point, used by main.py and the notebooks.

    If MANUAL_FEATURES is set in .env, that list is used directly instead
    of running automated selection. This is useful after looking at
    plots/feature_correlation_heatmap.png and deciding on your own columns.
    Otherwise this falls back to select_k_best_features above.

    Either way, the return value has the same shape (source, best_k,
    scores_by_k, selected_features, features_by_k), so callers do not need
    to know which path was taken.

    features_by_k[6] is always filled in, because the QCNN model needs
    exactly 6 qubits. It comes from MANUAL_FEATURES_QCNN (.env) if that is
    set (independently of MANUAL_FEATURES, which covers every other model),
    from MANUAL_FEATURES itself when that happens to already be a 6 feature
    list, or otherwise from an automated 6 feature selection.
    """
    from src.config import MANUAL_FEATURES, MANUAL_FEATURES_QCNN

    if MANUAL_FEATURES:
        invalid = [f for f in MANUAL_FEATURES if f not in X.columns]
        if invalid:
            raise ValueError(
                f"MANUAL_FEATURES (.env) contains unknown column(s): {invalid}. "
                f"Valid candidates are: {list(X.columns)}"
            )
        logger.info("Using manual feature override from .env: %s", MANUAL_FEATURES)

        features_by_k = {len(MANUAL_FEATURES): list(MANUAL_FEATURES)}
        if MANUAL_FEATURES_QCNN:
            features_by_k[6] = _resolve_qcnn_features(X, y, MANUAL_FEATURES_QCNN)
        elif len(MANUAL_FEATURES) != 6:
            logger.info(
                "MANUAL_FEATURES has %d feature(s), not 6, so the QCNN model "
                "(which needs exactly 6 qubits) will use an automated 6 feature selection instead.",
                len(MANUAL_FEATURES),
            )
            features_by_k[6] = _resolve_qcnn_features(X, y, None)

        return {
            "source": "manual",
            "best_k": len(MANUAL_FEATURES),
            "scores_by_k": {},
            "selected_features": list(MANUAL_FEATURES),
            "features_by_k": features_by_k,
        }

    result = select_k_best_features(X, y, k_range=k_range, **kwargs)
    result["source"] = "automated"
    if MANUAL_FEATURES_QCNN:
        result["features_by_k"][6] = _resolve_qcnn_features(X, y, MANUAL_FEATURES_QCNN)
    return result


def compute_feature_correlations(data: pd.DataFrame, target_col: str = REGRESSION_TARGET) -> pd.Series:
    """Pearson correlation of every candidate feature with the given target."""
    X = get_candidate_matrix(data)
    return X.apply(lambda col: col.corr(data[target_col])).sort_values(key=np.abs, ascending=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from src.data_fetching import load_or_fetch_smiles

    smiles_df = load_or_fetch_smiles()
    descriptors = build_descriptor_table(smiles_df)
    data = build_modeling_table(descriptors)

    X = get_candidate_matrix(data)
    y = data[CLASSIFICATION_TARGET]
    result = select_k_best_features(X, y)
    logger.info("Best k: %d, scores: %s", result["best_k"], result["scores_by_k"])
    logger.info("Selected features: %s", result["selected_features"])
