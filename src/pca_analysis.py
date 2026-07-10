"""PCA diagnostic on the candidate descriptor set (Task 1 supplement).

PCA here is explicitly NOT a feature-reduction step fed into the quantum/
classical models -- it is unsupervised (maximises Var(X), not Cov(X, y)),
and every downstream model must keep physically-interpretable RDKit
descriptors as its features, not abstract PCA components. Its job is
diagnostic: reveal *redundancy* in the 15-column candidate pool (e.g. the
"size cluster" -- MolWt, Chi0v, Chi1v, Kappa1-3, BertzCT -- collapsing onto
one latent axis).

That redundancy signal is then used to make the automated feature selector
in `features.py::select_k_best_features` cluster-aware: instead of blindly
taking the top-k univariate scorers (which tends to pick several redundant
descriptors from the same latent cluster), it takes at most one
representative per PCA-identified cluster before falling back to raw score.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def run_pca(X: pd.DataFrame, n_components: int | None = None) -> tuple[PCA, np.ndarray, pd.DataFrame]:
    """Standardize X and fit PCA. Returns (fitted PCA, scores, loadings).

    Standardization is essential: MolWt (~300-600), NumHDonors (0-4), and
    BertzCT (~400-1200) live on wildly different scales -- without scaling,
    high-variance features would dominate every PC purely from their units,
    not from any real structure.

    `loadings` is features x PCs, scaled by sqrt(explained_variance) so
    arrow/bar magnitudes are directly comparable across components.
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
    """Assign each feature to its "dominant" principal component: the PC
    (among the first `n_top_pcs`, which carry most of the variance) on
    which that feature has the largest absolute loading. Features sharing a
    dominant PC are, by construction, capturing largely the same latent
    axis of variation -- i.e. they are redundant with each other.
    """
    top = loadings.iloc[:, :min(n_top_pcs, loadings.shape[1])]
    return {feat: int(np.argmax(np.abs(top.loc[feat].values))) for feat in top.index}


def redundancy_aware_topk(
    scores: np.ndarray, feature_names: np.ndarray, clusters: dict[str, int], k: int
) -> list[str]:
    """Pick k features from `feature_names`, ranked by `scores` descending,
    but preferring at most one representative per PCA cluster: the
    highest-scoring feature from each cluster is taken first (in score
    order across clusters), and only once every cluster has contributed one
    feature does a second feature from any cluster get considered.

    This directly counters the failure mode of plain top-k univariate
    selection on this dataset: picking several highly-collinear descriptors
    (e.g. Kappa1, Kappa2, Kappa3, Chi0v -- all loading on the same "size"
    PC) that add little independent signal over one of them alone.
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
