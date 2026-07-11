"""Entry point for the whole pipeline. Run this to fetch data, build
features, train the classical and quantum models, and generate every plot
and results table.

Usage:
    python main.py --stage all
    python main.py --stage fetch
    python main.py --stage features
    python main.py --stage classical                  classical classifiers + PLS regression
    python main.py --stage classical-classification    classical classifiers only (SVC/RandomForest/GradientBoosting)
    python main.py --stage classical-regression        PLS regression baseline only
    python main.py --stage quantum                     quantum classifiers + quantum regressors
    python main.py --stage quantum-classification       quantum classifiers only (QK-SVM/VQC/QCNN)
    python main.py --stage quantum-regression            quantum regressors only (QK-KRR/VQR/QCNN-R)
    python main.py --stage bonus

Every quantum circuit runs for real through a Sampler job (see
src/quantum_backend.py), never as a Statevector shortcut. Choose where with
--backend:
    python main.py --backend aer            local AerSimulator, no noise (default)
    python main.py --backend aer-noisy       local AerSimulator with a device-like noise model
    python main.py --backend ibm-runtime     a real IBM Quantum backend
        (run python scripts/setup_ibm_account.py first, see the README)

    python main.py --backend ibm-runtime --max-samples 8
        a cheap smoke test to run before committing to a full 29 fold LOOCV
        run on a real, queued device.
"""

import argparse
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.bonus_extensions import (
    BlindPredictor,
    KTAOptimizedQuantumKernel,
    execution_degradation_study,
    fit_deployment_model,
)
from src.classical_models import run_classical_classification_suite, run_pls_regression_baseline
from src.config import (
    CLASSIFICATION_TARGET,
    DATASET_CSV,
    DESCRIPTORS_CSV,
    MODELING_TABLE_CSV,
    PATZMANN_Q2_BENCHMARK,
    PLOTS_DIR,
    REGRESSION_TARGET,
    RESULTS_DIR,
)
from src.data_fetching import load_or_fetch_smiles
from src.evaluation import kernel_target_alignment, save_results_table, summarize_results
from src.features import (
    build_descriptor_table,
    build_modeling_table,
    compute_feature_correlations,
    get_candidate_matrix,
    get_modeling_features,
)
from src.pca_analysis import get_pc_dominant_cluster, run_pca
from src.quantum_backend import ExecutionConfig, QuantumExecutor
from src.quantum_circuits import build_regression_circuit, build_vqc_circuit, reuploading_layer
from src.quantum_models import (
    FEATURE_MAPS,
    QUANTUM_MODEL_BUILDERS,
    QUANTUM_REGRESSION_MODEL_BUILDERS,
    build_qcnn_circuit,
    compute_quantum_kernel_matrix,
    run_quantum_classification_suite,
    run_quantum_regression_suite,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
# qiskit's transpiler and qiskit-ibm-runtime's primitive layer both log a
# lot of detail at INFO level, which drowns out this pipeline's own
# progress messages. Keep those libraries quieter.
for _noisy_logger in ("qiskit", "qiskit_ibm_runtime", "qiskit_aer", "stevedore"):
    logging.getLogger(_noisy_logger).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

BACKEND_MODE_MAP = {"aer": "aer_simulator", "aer-noisy": "aer_noisy", "ibm-runtime": "ibm_runtime"}


# --- Plotting helpers ---------------------------------------------------------

def plot_feature_correlation_heatmap(X_full: pd.DataFrame, data: pd.DataFrame) -> None:
    """Draw the Pearson correlation heatmap for every candidate feature
    plus the target: the full square matrix, a diverging colormap centered
    at zero, the value printed in every cell, and a colorbar.
    """
    corr_matrix = pd.concat([X_full, data[[REGRESSION_TARGET]]], axis=1).corr()

    n = len(corr_matrix)
    fig, ax = plt.subplots(figsize=(max(10, n * 0.85), max(8.5, n * 0.75)))
    sns.heatmap(
        corr_matrix,
        cmap="RdBu_r",
        vmin=-1, vmax=1, center=0,
        annot=True, fmt=".2f", annot_kws={"size": 8},
        linewidths=0.6, linecolor="white",
        square=True, ax=ax,
        cbar_kws={"label": "Pearson correlation (r)", "shrink": 0.8},
    )
    ax.set_title(
        "Feature Correlation Matrix\n14 RDKit descriptors + D50 + apparent_solubility + COMDR_15min (target)",
        fontsize=13, fontweight="bold", pad=14,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

    # Bold the target's row and column labels so it stands out.
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        if label.get_text() == REGRESSION_TARGET:
            label.set_fontweight("bold")

    # Seaborn draws all annotation text in one fixed color by default,
    # which is hard to read on the dark end of a diverging colormap. Pick
    # white or black per cell based on how light or dark that cell is.
    cmap = plt.get_cmap("RdBu_r")
    norm = plt.Normalize(vmin=-1, vmax=1)
    for text, value in zip(ax.texts, corr_matrix.values.flatten()):
        r, g, b, _ = cmap(norm(value))
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        text.set_color("white" if luminance < 0.55 else "black")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "feature_correlation_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_pca_diagnostics(X_full: pd.DataFrame, data: pd.DataFrame) -> tuple:
    """Draw the PCA diagnostic plots for Task 1: a scree plot, a PC1 vs PC2
    scatter colored by Responder or Non-Responder, a loadings biplot, and
    each component's correlation with the continuous target.

    See pca_analysis.py for why PCA is used as a diagnostic here rather
    than a feature reduction step. Returns the fitted PCA object, scores,
    and loadings so the caller can also save the numeric tables and report
    the redundancy clusters that feed into feature selection.
    """
    pca, scores, loadings = run_pca(X_full)
    pca_dir = PLOTS_DIR / "pca"

    ev = pca.explained_variance_ratio_ * 100
    cum = np.cumsum(ev)
    n80 = int(np.argmax(cum >= 80) + 1)
    n95 = int(np.argmax(cum >= 95) + 1)

    # 1. Scree plot: variance per component as bars, cumulative variance as a line.
    k = np.arange(1, len(ev) + 1)
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.bar(k, ev, color="#2471a3", alpha=0.85, label="per-PC variance")
    ax1.set_xlabel("Principal component")
    ax1.set_ylabel("Variance explained (%)")
    ax1.set_xticks(k)
    ax2 = ax1.twinx()
    ax2.plot(k, cum, "o-", color="#c0392b", lw=2, label="cumulative")
    ax2.axhline(80, color="grey", ls="--", lw=1)
    ax2.set_ylabel("Cumulative variance (%)")
    ax2.set_ylim(0, 105)
    ax1.set_title(f"PCA scree: {n80} PC(s) cover 80% variance, {n95} cover 95%")
    fig.tight_layout()
    fig.savefig(pca_dir / "pca_scree.png", dpi=150)
    plt.close(fig)

    # 2. PC1 vs PC2 scatter, colored by responder or non-responder, sized by COMDR_15min.
    resp = data[CLASSIFICATION_TARGET].values == 1
    target = data[REGRESSION_TARGET].values
    fig, ax = plt.subplots(figsize=(7.2, 6))
    sizes = 30 + 8 * np.clip(target, None, 15)
    ax.scatter(scores[~resp, 0], scores[~resp, 1], s=sizes[~resp], c="#c0392b",
               edgecolor="k", lw=0.5, alpha=0.85, label="non-responder")
    ax.scatter(scores[resp, 0], scores[resp, 1], s=sizes[resp], c="#2471a3",
               edgecolor="k", lw=0.5, alpha=0.85, label="responder")
    for i in np.argsort(-target)[:4]:
        ax.annotate(data["drug"].iloc[i], (scores[i, 0], scores[i, 1]),
                    fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.axhline(0, color="k", lw=0.4)
    ax.axvline(0, color="k", lw=0.4)
    ax.set_xlabel(f"PC1 ({ev[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({ev[1]:.1f}% variance)")
    ax.set_title("Drugs in PC1-PC2 space (marker size ~ COMDR_15min)")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(pca_dir / "pca_scatter.png", dpi=150)
    plt.close(fig)

    # 3. Biplot: feature loading arrows plus drug scores, both on PC1 and PC2.
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = scores[:, :2] / np.max(np.abs(scores[:, :2]))
    ld = loadings[["PC1", "PC2"]].values
    ld = ld / np.max(np.abs(ld)) * 0.9
    ax.scatter(sc[~resp, 0], sc[~resp, 1], s=45, c="#c0392b", alpha=0.7, edgecolor="k", lw=0.4, label="non-responder")
    ax.scatter(sc[resp, 0], sc[resp, 1], s=45, c="#2471a3", alpha=0.7, edgecolor="k", lw=0.4, label="responder")
    for i, feat in enumerate(loadings.index):
        ax.arrow(0, 0, ld[i, 0], ld[i, 1], color="k", alpha=0.6, head_width=0.02, length_includes_head=True)
        ax.text(ld[i, 0] * 1.12, ld[i, 1] * 1.12, feat, fontsize=8, ha="center", va="center", color="k",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
    ax.axhline(0, color="k", lw=0.4)
    ax.axvline(0, color="k", lw=0.4)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xlabel(f"PC1 ({ev[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1]:.1f}%)")
    ax.set_title("Biplot: feature loadings (arrows) + drug scores (dots)")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(pca_dir / "pca_biplot.png", dpi=150)
    plt.close(fig)

    # 4. Each component's correlation with the continuous target, Pearson and Spearman.
    npcs = min(6, scores.shape[1])
    corrs_p = [np.corrcoef(scores[:, kk], target)[0, 1] for kk in range(npcs)]
    corrs_s = [pd.Series(scores[:, kk]).corr(pd.Series(target), method="spearman") for kk in range(npcs)]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(npcs)
    ax.bar(x - 0.2, corrs_p, width=0.4, color="#2471a3", label="Pearson")
    ax.bar(x + 0.2, corrs_s, width=0.4, color="#c0392b", label="Spearman")
    for kk in range(npcs):
        ax.text(kk, max(corrs_p[kk], corrs_s[kk]) + 0.02, f"{ev[kk]:.0f}%", ha="center", fontsize=8, color="grey")
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"PC{i + 1}" for i in range(npcs)])
    ax.set_ylabel("correlation with COMDR_15min")
    ax.set_ylim(-1, 1)
    ax.set_title("Are the top PCs predictive? (grey label = variance explained)")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(pca_dir / "pca_target_corr.png", dpi=150)
    plt.close(fig)

    logger.info("Saved 4 PCA diagnostic plots to %s (%d PC(s) cover 80%% variance, %d cover 95%%)", pca_dir, n80, n95)
    return pca, scores, loadings


# --- Stage 1: data + features ------------------------------------------------

def stage_features() -> tuple[pd.DataFrame, dict]:
    smiles_df = load_or_fetch_smiles()
    if DESCRIPTORS_CSV.exists():
        descriptors = pd.read_csv(DESCRIPTORS_CSV)
    else:
        descriptors = build_descriptor_table(smiles_df)
    data = build_modeling_table(descriptors, dataset_path=DATASET_CSV)

    X_full = get_candidate_matrix(data)
    y_clf = data[CLASSIFICATION_TARGET]

    corr = compute_feature_correlations(data)
    logger.info("Top correlations with COMDR_15min:\n%s", corr.to_string())

    plot_feature_correlation_heatmap(X_full, data)

    pca, pca_scores, pca_loadings = plot_pca_diagnostics(X_full, data)
    pca_loadings.round(4).to_csv(RESULTS_DIR / "pca_loadings.csv")
    pd.DataFrame(
        pca_scores, columns=pca_loadings.columns, index=data["drug"]
    ).round(4).to_csv(RESULTS_DIR / "pca_scores.csv")
    clusters = get_pc_dominant_cluster(pca_loadings)
    cluster_groups: dict[int, list[str]] = {}
    for feat, pc in clusters.items():
        cluster_groups.setdefault(pc, []).append(feat)
    logger.info(
        "PCA redundancy clusters (features sharing a dominant PC among the top 3): %s",
        {f"PC{pc + 1}": feats for pc, feats in sorted(cluster_groups.items())},
    )

    selection = get_modeling_features(X_full, y_clf)
    logger.info(
        "Feature selection (source=%s): best_k=%d, scores_by_k=%s",
        selection["source"], selection["best_k"], selection["scores_by_k"],
    )
    logger.info("Selected features: %s", selection["selected_features"])
    logger.info("6-feature subset (for the fixed 6-qubit QCNN): %s", selection["features_by_k"][6])

    pd.Series(selection["selected_features"]).to_csv(RESULTS_DIR / "selected_features.csv", index=False, header=["feature"])
    return data, selection


# --- Stage 2: classical baselines --------------------------------------------

def stage_classical_classification(data: pd.DataFrame, selection: dict) -> dict:
    """Classical classifiers only (SVC/RandomForest/GradientBoosting) --
    runnable standalone via `--stage classical-classification`.
    """
    selected_features = selection["selected_features"]
    X = data[selected_features]
    y_clf = data[CLASSIFICATION_TARGET]

    clf_results = run_classical_classification_suite(X, y_clf)
    clf_table = summarize_results(clf_results, kind="classification")
    save_results_table(clf_table, "classical_loocv.csv")
    logger.info("Classical LOOCV results:\n%s", clf_table.to_string())
    return clf_results


def stage_classical_regression(data: pd.DataFrame, selection: dict) -> dict:
    """PLS regression baseline only -- runnable standalone via
    `--stage classical-regression`.
    """
    selected_features = selection["selected_features"]
    X = data[selected_features]
    y_reg = data[REGRESSION_TARGET]

    pls_result = run_pls_regression_baseline(X, y_reg)
    pls_table = pd.DataFrame([{
        "model": "PLS_regression",
        "features": ", ".join(selected_features),
        "n_components": pls_result["metrics"]["n_components"],
        "target": pls_result["metrics"]["target_scale"],
        "q2_loocv": round(pls_result["metrics"]["q2_loocv"], 4),
        "r2_train": round(pls_result["metrics"]["r2_train"], 4),
        "patzmann_r2_benchmark": 0.82,
        "patzmann_q2_benchmark": PATZMANN_Q2_BENCHMARK,
        "n_folds": pls_result["metrics"]["n_folds"],
    }])
    save_results_table(pls_table, "pls_regression_loocv.csv")
    logger.info(
        "PLS regression LOOCV: Q2=%.3f (Patzmann benchmark Q2=%.2f), R2(train)=%.3f (Patzmann R2=0.82)",
        pls_result["metrics"]["q2_loocv"], PATZMANN_Q2_BENCHMARK, pls_result["metrics"]["r2_train"],
    )
    return pls_result


def stage_classical(data: pd.DataFrame, selection: dict) -> dict:
    """Both classical stages together -- used by `--stage classical` / `all`."""
    clf_results = stage_classical_classification(data, selection)
    stage_classical_regression(data, selection)
    return clf_results


# --- Circuit diagrams (Task 3 deliverable) -----------------------------------

def plot_circuit_diagrams(n_features: int, n_qcnn_features: int = 6) -> None:
    """Render one diagram per distinct circuit shape used across every
    registered quantum model (classification + regression), each saved
    under a filename that names the model(s) that use it.

    `zz_feature_map` / `entangled_feature_map` stay in the Task 3 kernel
    comparison below even though no classifier currently uses them by
    that name, since that comparison is about encoding geometry, not
    which model is registered.
    """
    angle_circ = FEATURE_MAPS["angle"](n_features)[0]
    entangled_circ = FEATURE_MAPS["entangled"](n_features)[0]
    zz_circ = FEATURE_MAPS["zz"](n_features)[0]
    zzplus_circ = FEATURE_MAPS["zzplus"](n_features)[0]
    vqc_circ = build_vqc_circuit(n_features, n_layers=2)[0]
    # QK-SVM_trained's own kernel feature map (n_layers=3 reuploading_layer).
    reuploading_circ = reuploading_layer(n_features, n_layers=3)[0]
    vqr_circ = build_regression_circuit(n_features, n_layers=1, n_output_qubits=3)[0]
    qcnn_circ = build_qcnn_circuit(n_qcnn_features)[0]

    circuits = {
        # Feature maps (Task 3 comparison).
        "angle_feature_map": angle_circ,
        "entangled_feature_map": entangled_circ,
        "zz_feature_map": zz_circ,
        "zzplus_feature_map": zzplus_circ,
        # Ansatze / full model circuits not already covered above.
        "vqc_ansatz": vqc_circ,
        "vqr_circuit": vqr_circ,
        "qcnn": qcnn_circ,
        # Explicit per-model aliases -- QK-SVM's fixed vs. trained kernel,
        # and every other registered model that would otherwise have no
        # image discoverable under its own name.
        "qk_svm_angle_kernel": angle_circ,
        "qk_svm_trained_kernel": reuploading_circ,
        "qk_krr_angle_kernel": angle_circ,
        "qcnn_regressor": qcnn_circ,
    }
    for name, circ in circuits.items():
        fig = circ.draw("mpl", fold=-1)
        fig.savefig(PLOTS_DIR / "circuits" / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    logger.info("Saved %d circuit diagrams to %s", len(circuits), PLOTS_DIR / "circuits")


def plot_quantum_regression_q2(reg_table: pd.DataFrame) -> None:
    """Bar chart of each quantum regressor's LOOCV Q^2 against the Patzmann
    et al. Q^2=0.77 benchmark line -- the regression-suite counterpart of
    `stage_unified_comparison`'s classification accuracy bar chart.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#55A868" if q2 >= 0 else "#C44E52" for q2 in reg_table["q2"]]
    ax.bar(reg_table["model"], reg_table["q2"], color=colors)
    ax.axhline(
        PATZMANN_Q2_BENCHMARK, color="#4C72B0", linestyle="--", linewidth=1.5,
        label=f"Patzmann Q2 benchmark ({PATZMANN_Q2_BENCHMARK:.2f})",
    )
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_ylabel("Q2 (LOOCV)")
    ax.set_title("Quantum regression suite vs. Patzmann Q2 benchmark")
    plt.xticks(rotation=20, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "scatter" / "quantum_regression_q2.png", dpi=150)
    plt.close(fig)


# --- Stage 3: quantum feature map + models -----------------------------------

def _sliced_features_and_targets(data: pd.DataFrame, selection: dict, max_samples: int | None):
    """Shared setup for both quantum sub-stages: the standard
    selected-feature matrix, the dedicated 6-feature QCNN subset, and both
    targets, all consistently sliced by `max_samples` (a smoke-test cap,
    not the full Task 4 deliverable, when set below 29).
    """
    selected_features = selection["selected_features"]
    X = data[selected_features].values
    X_qcnn = data[selection["features_by_k"][6]].values
    y_clf = data[CLASSIFICATION_TARGET].values
    y_reg = data[REGRESSION_TARGET].values

    if max_samples is not None and max_samples < len(X):
        logger.warning(
            "max_samples=%d < %d: running a REDUCED, non-full LOOCV for practicality on backend unset here. "
            "Results are a smoke test, not the full deliverable.",
            max_samples, len(X),
        )
        X, X_qcnn, y_clf, y_reg = X[:max_samples], X_qcnn[:max_samples], y_clf[:max_samples], y_reg[:max_samples]
    return X, X_qcnn, y_clf, y_reg


def stage_quantum_classification(
    data: pd.DataFrame, selection: dict, execution_config: ExecutionConfig, max_samples: int | None
) -> dict:
    """Quantum classifiers only (Task 3 kernel/KTA diagnostics + Task 4
    classification LOOCV) -- runnable standalone via
    `--stage quantum-classification`.
    """
    from sklearn.preprocessing import MinMaxScaler

    selected_features = selection["selected_features"]
    X, X_qcnn, y, _ = _sliced_features_and_targets(data, selection, max_samples)

    plot_circuit_diagrams(n_features=len(selected_features), n_qcnn_features=len(selection["features_by_k"][6]))

    # Task 3: kernel heatmap and KTA score for each feature map, sharing one executor.
    kernel_executor = QuantumExecutor(execution_config)
    for fm_name in ["angle", "entangled", "zz"]:
        X_scaled = MinMaxScaler(feature_range=(0, np.pi)).fit_transform(X)
        K = compute_quantum_kernel_matrix(X_scaled, kernel_executor, feature_map_name=fm_name)
        kta = kernel_target_alignment(K, y)
        logger.info("Feature map=%s | KTA=%.4f", fm_name, kta)

        order = np.argsort(y)
        K_sorted = K[np.ix_(order, order)]
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(K_sorted, cmap="viridis", ax=ax, square=True)
        ax.set_title(f"Quantum kernel matrix ({fm_name}, {execution_config.label()}) sorted by label | KTA={kta:.3f}")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "kernels" / f"kernel_heatmap_{fm_name}.png", dpi=150)
        plt.close(fig)

    # Task 4: run the full quantum classifier suite under LOOCV. The QCNN
    # needs exactly 6 qubits, so it runs separately on its own 6 feature
    # subset instead of whatever k the automated selector picked for the others.
    non_qcnn_names = [n for n in QUANTUM_MODEL_BUILDERS if n != "QCNN"]
    quantum_results = run_quantum_classification_suite(X, y, execution_config=execution_config, model_names=non_qcnn_names)

    logger.info("Running QCNN on its dedicated 6-feature subset: %s", selection["features_by_k"][6])
    quantum_results.update(
        run_quantum_classification_suite(X_qcnn, y, execution_config=execution_config, model_names=["QCNN"])
    )

    q_table = summarize_results(quantum_results, kind="classification")
    save_results_table(q_table, "quantum_loocv.csv")
    logger.info("Quantum LOOCV results (backend=%s):\n%s", execution_config.label(), q_table.to_string())

    # Plot each drug's predicted score against its true COMDR_15min, using the best quantum model.
    best_name = q_table.iloc[0]["model"]
    best_res = quantum_results[best_name]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(data[REGRESSION_TARGET].values[: len(best_res["y_true"])], best_res["y_proba"],
               c=best_res["y_true"], cmap="coolwarm")
    ax.set_xlabel("True COMDR_15min")
    ax.set_ylabel(f"P(|1>) from {best_name}")
    ax.set_title(f"Per-drug quantum score vs. true COMDR_15min ({best_name})")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "scatter" / "score_vs_comdr.png", dpi=150)
    plt.close(fig)

    return quantum_results


def stage_quantum_regression(
    data: pd.DataFrame, selection: dict, execution_config: ExecutionConfig, max_samples: int | None
) -> dict:
    """Quantum regressors only (QK-KRR, VQR, QCNN-R),
    predicting COMDR_15min directly -- comparable to the Patzmann
    Q^2=0.77 / R^2=0.82 benchmark. Runnable standalone via
    `--stage quantum-regression`. QCNN-R shares QCNN's fixed-6-qubit
    constraint, so it runs on the dedicated 6-feature subset.
    """
    selected_features = selection["selected_features"]
    X, X_qcnn, _, y_reg = _sliced_features_and_targets(data, selection, max_samples)

    plot_circuit_diagrams(n_features=len(selected_features), n_qcnn_features=len(selection["features_by_k"][6]))

    non_qcnn_reg_names = [n for n in QUANTUM_REGRESSION_MODEL_BUILDERS if n != "QCNN-R"]
    quantum_regression_results = run_quantum_regression_suite(
        X, y_reg, execution_config=execution_config, model_names=non_qcnn_reg_names
    )
    logger.info("Running QCNN-R on its dedicated 6-feature subset: %s", selection["features_by_k"][6])
    quantum_regression_results.update(
        run_quantum_regression_suite(X_qcnn, y_reg, execution_config=execution_config, model_names=["QCNN-R"])
    )

    reg_table = summarize_results(quantum_regression_results, kind="regression")
    save_results_table(reg_table, "quantum_regression_loocv.csv")
    logger.info(
        "Quantum regression LOOCV results (backend=%s, Patzmann Q2 benchmark=%.2f):\n%s",
        execution_config.label(), PATZMANN_Q2_BENCHMARK, reg_table.to_string(),
    )
    plot_quantum_regression_q2(reg_table)
    return quantum_regression_results


def stage_quantum(data: pd.DataFrame, selection: dict, execution_config: ExecutionConfig, max_samples: int | None) -> dict:
    """Both quantum stages together -- used by `--stage quantum` / `all`."""
    quantum_results = stage_quantum_classification(data, selection, execution_config, max_samples)
    stage_quantum_regression(data, selection, execution_config, max_samples)
    return quantum_results


# --- Stage 4: bonus extensions -----------------------------------------------

def stage_bonus(data: pd.DataFrame, selection: dict, execution_config: ExecutionConfig, max_samples: int | None) -> None:
    from sklearn.preprocessing import MinMaxScaler

    selected_features = selection["selected_features"]
    X = data[selected_features].values
    y = data[CLASSIFICATION_TARGET].values
    if max_samples is not None and max_samples < len(X):
        X, y = X[:max_samples], y[:max_samples]

    logger.info("Running KTA optimization (trained quantum kernel, backend=%s)...", execution_config.label())
    X_scaled = MinMaxScaler(feature_range=(0, np.pi)).fit_transform(X)
    kta_executor = QuantumExecutor(execution_config)
    kta_model = KTAOptimizedQuantumKernel(executor=kta_executor, n_layers=3, maxiter=60)
    kta_model.fit(X_scaled, y)
    logger.info("KTA optimization: %.4f -> %.4f", kta_model.kta_before_, kta_model.kta_after_)

    logger.info("Running execution-degradation study (ideal vs. %s)...", execution_config.label())
    comparison_config = execution_config if execution_config.mode != "aer_simulator" else ExecutionConfig(mode="aer_noisy")
    degradation_result = execution_degradation_study(
        X, y, feature_map_name="angle", n_splits=min(5, min(np.bincount(y.astype(int)))),
        comparison_config=comparison_config,
    )
    logger.info(
        "Degradation study: baseline(%s)=%.3f comparison(%s)=%.3f degradation=%.3f",
        degradation_result["baseline_label"], degradation_result["baseline_accuracy_mean"],
        degradation_result["comparison_label"], degradation_result["comparison_accuracy_mean"],
        degradation_result["degradation"],
    )
    pd.DataFrame([degradation_result]).to_csv(RESULTS_DIR / "execution_degradation.csv", index=False)

    logger.info("Fitting deployment model + testing blind SMILES interface...")
    from src.classical_models import make_svc

    model, scaler = fit_deployment_model(make_svc, data[selected_features], data[CLASSIFICATION_TARGET].values)
    blind = BlindPredictor(model, scaler, selected_features)
    demo_smiles = {"Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O", "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"}
    for name, smiles in demo_smiles.items():
        kwargs = {"d50": 50.0} if "D50" in selected_features else {}
        result = blind.predict(smiles, **kwargs)
        logger.info("Blind prediction | %s -> %s (confidence=%.3f)", name, result["prediction"], result["confidence"])


# --- Stage 5: unified comparison ---------------------------------------------

def stage_unified_comparison(clf_results: dict, quantum_results: dict) -> None:
    all_results = {**clf_results, **quantum_results}
    table = summarize_results(all_results, kind="classification")
    save_results_table(table, "unified_comparison.csv")
    logger.info("Unified comparison:\n%s", table.to_string())

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#4C72B0" if m in clf_results else "#DD8452" for m in table["model"]]
    ax.bar(table["model"], table["accuracy"], color=colors)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1, label="majority baseline")
    ax.set_ylabel("LOOCV accuracy")
    ax.set_title("Classical (blue) vs Quantum (orange): unified LOOCV comparison")
    plt.xticks(rotation=30, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "unified_comparison.png", dpi=150)
    plt.close(fig)


STAGE_CHOICES = [
    "fetch",
    "features",
    "classical",
    "classical-classification",
    "classical-regression",
    "quantum",
    "quantum-classification",
    "quantum-regression",
    "bonus",
    "all",
]


def main():
    parser = argparse.ArgumentParser(description="OQI Hackathon 2026 QML pipeline")
    parser.add_argument("--stage", choices=STAGE_CHOICES, default="all")
    parser.add_argument(
        "--backend", choices=list(BACKEND_MODE_MAP), default="aer",
        help="Where quantum circuits actually execute: local ideal (aer), local noisy (aer-noisy), or real IBM Quantum hardware (ibm-runtime).",
    )
    parser.add_argument("--shots", type=int, default=4096, help="Shots per circuit execution.")
    parser.add_argument("--ibm-backend", type=str, default=None, help="Specific IBM backend name (default: least-busy).")
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Cap the number of drugs used in the quantum/bonus stages (< 29 = reduced, non-full LOOCV). "
             "Strongly recommended when --backend ibm-runtime, to bound queue time/cost before a full run.",
    )
    args = parser.parse_args()

    execution_config = ExecutionConfig(
        mode=BACKEND_MODE_MAP[args.backend], shots=args.shots, ibm_backend_name=args.ibm_backend,
    )

    if args.stage == "fetch":
        load_or_fetch_smiles()
        return

    data, selection = stage_features()

    if args.stage == "features":
        return

    clf_results = None
    quantum_results = None

    if args.stage == "classical-classification":
        clf_results = stage_classical_classification(data, selection)
    elif args.stage == "classical-regression":
        stage_classical_regression(data, selection)
    elif args.stage in ("classical", "all"):
        clf_results = stage_classical(data, selection)

    if args.stage == "quantum-classification":
        quantum_results = stage_quantum_classification(data, selection, execution_config, args.max_samples)
    elif args.stage == "quantum-regression":
        stage_quantum_regression(data, selection, execution_config, args.max_samples)
    elif args.stage in ("quantum", "all"):
        quantum_results = stage_quantum(data, selection, execution_config, args.max_samples)

    if args.stage in ("bonus", "all"):
        stage_bonus(data, selection, execution_config, args.max_samples)
    if args.stage == "all" and clf_results is not None and quantum_results is not None:
        stage_unified_comparison(clf_results, quantum_results)

    logger.info("Pipeline stage '%s' complete (backend=%s).", args.stage, execution_config.label())


if __name__ == "__main__":
    main()
