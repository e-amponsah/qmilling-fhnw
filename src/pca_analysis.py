"""PCA diagnostic on the candidate descriptor set.

This is not a feature reduction step. Nothing downstream trains on the
components themselves, every model keeps the original RDKit descriptors
as its input. What PCA is used for here is finding redundancy in the 16
candidate columns, since several of them (MolWt, Chi0v, Chi1v, the three
Kappas, BertzCT) mostly describe molecule size and collapse onto one
component. That redundancy feeds into the feature selector in features.py,
which spreads its picks across clusters instead of grabbing several
correlated descriptors from the same one.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def run_pca(X: pd.DataFrame, n_components: int | None = None) -> tuple[PCA, np.ndarray, pd.DataFrame]:
    """Standardize X and fit PCA. Returns the fitted PCA object, the scores, and the loadings.

    Standardization matters because the descriptors sit on very different
    scales (MolWt in the hundreds, NumHDonors a small integer, BertzCT in
    the thousands). Without it, the largest-magnitude columns would
    dominate every component just from their units, not real structure.
    """
    feats = list(X.columns)
    n_components = n_components or min(len(feats), len(X) - 1)

    Z = StandardScaler().fit_transform(X.values)
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(Z)

    loadings = pd.DataFrame(
        pca.components_.T * np.sqrt(pca.explained_variance_),
        index=feats,
        columns=[f"PC{i + 1}" for i in range(pca.n_components_)],
    )
    return pca, scores, loadings


def get_pc_dominant_cluster(loadings: pd.DataFrame, n_top_pcs: int = 3) -> dict[str, int]:
    """Assign each feature to the principal component it loads on most
    strongly, looking only at the first n_top_pcs components (the ones
    that carry most of the variance). Features that share a dominant
    component are measuring roughly the same underlying thing.
    """
    top = loadings.iloc[:, :min(n_top_pcs, loadings.shape[1])]
    return {feat: int(np.argmax(np.abs(top.loc[feat].values))) for feat in top.index}


def redundancy_aware_topk(
    scores: np.ndarray, feature_names: np.ndarray, clusters: dict[str, int], k: int
) -> list[str]:
    """Pick k features ranked by score, but spread the picks across PCA
    clusters instead of just taking the top k. The best feature from each
    cluster is taken first, and a second feature from the same cluster
    only gets considered once every cluster has contributed one. This
    keeps highly correlated descriptors like Kappa1, Kappa2, Kappa3 and
    Chi0v, which all load on the same size related component, from filling
    up the whole selection.
    """
    order = np.argsort(-scores)
    ranked_features = [feature_names[i] for i in order]

    selected: list[str] = []
    used_clusters: set[int] = set()
    for feat in ranked_features:
        if len(selected) == k:
            break
        cluster = clusters.get(feat)
        if cluster not in used_clusters:
            selected.append(feat)
            used_clusters.add(cluster)

    if len(selected) < k:
        for feat in ranked_features:
            if len(selected) == k:
                break
            if feat not in selected:
                selected.append(feat)

    return selected
