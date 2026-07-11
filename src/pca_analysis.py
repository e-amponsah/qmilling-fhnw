"""PCA diagnostic on the candidate descriptor set (Task 1 supplement).

This PCA is not a feature reduction step, and nothing downstream trains on
the PCA components themselves. PCA is unsupervised (it maximizes variance in
X, not covariance with the target y), and every model in this project keeps
physically interpretable RDKit descriptors as its input features.

What PCA is used for here is finding redundancy in the 16 column candidate
pool. For example, MolWt, Chi0v, Chi1v, Kappa1, Kappa2, Kappa3, and BertzCT
mostly describe the same thing (molecule size) and collapse onto a single
principal component.

That redundancy information then feeds into the feature selector in
features.py. Instead of blindly taking the top scoring descriptors, which
tends to pick several redundant ones from the same cluster, it takes at
most one feature per PCA cluster first.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def run_pca(X: pd.DataFrame, n_components: int | None = None) -> tuple[PCA, np.ndarray, pd.DataFrame]:
    """Standardize X and fit PCA. Returns the fitted PCA object, the scores, and the loadings.

    Standardization matters here because the raw descriptors live on very
    different scales (MolWt is in the hundreds, NumHDonors is a small
    integer, BertzCT is in the hundreds to thousands). Without scaling,
    the high magnitude columns would dominate every component just because
    of their units, not because they carry more real structure.

    The loadings table is features by components, scaled by the square
    root of the explained variance so that magnitudes are comparable
    across components.
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
    clusters instead of taking the top k regardless of cluster.

    The best scoring feature from each cluster is taken first, going
    through clusters in score order. Only once every cluster has
    contributed one feature does a second feature from the same cluster
    get considered. This avoids picking several highly correlated
    descriptors, such as Kappa1, Kappa2, Kappa3 and Chi0v, which all load
    on the same size related component and add little beyond what one of
    them already captures.
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
