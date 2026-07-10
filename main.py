"""Root execution gate steering the entire OQI Hackathon 2026 QML pipeline.

Usage:
    python main.py --stage all
    python main.py --stage fetch
    python main.py --stage features
    python main.py --stage classical
    python main.py --stage quantum
    python main.py --stage bonus

Backend toggle (every quantum circuit is executed for real via a Sampler
job -- see src/quantum_backend.py -- never a Statevector shortcut):
    python main.py --backend aer            # local AerSimulator, ideal (default)
    python main.py --backend aer-noisy       # local AerSimulator + device-like noise
    python main.py --backend ibm-runtime     # real IBM Quantum backend
        (requires `python scripts/setup_ibm_account.py` first; see README)

    python main.py --backend ibm-runtime --max-samples 8   # cheap smoke test
        before committing a full 29-fold LOOCV run to a real, queued device.
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
    select_k_best_features,
)
from src.quantum_backend import ExecutionConfig, QuantumExecutor
from src.quantum_circuits import build_vqc_circuit, reuploading_layer
from src.quantum_models import (
    FEATURE_MAPS,
    QUANTUM_MODEL_BUILDERS,
    build_qcnn_circuit,
    compute_quantum_kernel_matrix,
    run_quantum_classification_suite,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
# qiskit's transpiler and qiskit-ibm-runtime's primitive layer both log
# per-pass / per-job-submission detail at INFO, which drowns out this
# pipeline's own progress logging; keep them at WARNING.
for _noisy_logger in ("qiskit", "qiskit_ibm_runtime", "qiskit_aer", "stevedore"):
    logging.getLogger(_noisy_logger).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

BACKEND_MODE_MAP = {"aer": "aer_simulator", "aer-noisy": "aer_noisy", "ibm-runtime": "ibm_runtime"}


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

    fig, ax = plt.subplots(figsize=(8, 6))
    corr_matrix = pd.concat([X_full, data[[REGRESSION_TARGET]]], axis=1).corr()
    sns.heatmap(corr_matrix, cmap="RdBu_r", center=0, annot=False, ax=ax)
    ax.set_title("Feature correlation heatmap (14 RDKit descriptors + D50 + COMDR_15min)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "feature_correlation_heatmap.png", dpi=150)
    plt.close(fig)

    selection = select_k_best_features(X_full, y_clf)
    logger.info("Feature selection: best_k=%d, scores_by_k=%s", selection["best_k"], selection["scores_by_k"])
    logger.info("Selected features: %s", selection["selected_features"])
    logger.info("6-feature subset (for the fixed 6-qubit QCNN): %s", selection["features_by_k"][6])

    pd.Series(selection["selected_features"]).to_csv(RESULTS_DIR / "selected_features.csv", index=False, header=["feature"])
    return data, selection


# --- Stage 2: classical baselines --------------------------------------------

def stage_classical(data: pd.DataFrame, selection: dict) -> dict:
    selected_features = selection["selected_features"]
    X = data[selected_features]
    y_clf = data[CLASSIFICATION_TARGET]
    y_reg = data[REGRESSION_TARGET]

    clf_results = run_classical_classification_suite(X, y_clf)
    clf_table = summarize_results(clf_results, kind="classification")
    save_results_table(clf_table, "classical_loocv.csv")
    logger.info("Classical LOOCV results:\n%s", clf_table.to_string())

    pls_result = run_pls_regression_baseline(X, y_reg)
    logger.info(
        "PLS regression LOOCV: Q2=%.3f (Patzmann benchmark Q2=%.2f), R2(train)=%.3f",
        pls_result["metrics"]["q2_loocv"], PATZMANN_Q2_BENCHMARK, pls_result["metrics"]["r2_train"],
    )
    return clf_results


# --- Circuit diagrams (Task 3 deliverable) -----------------------------------

def plot_circuit_diagrams(n_features: int, n_qcnn_features: int = 6) -> None:
    circuits = {
        "angle_feature_map": FEATURE_MAPS["angle"](n_features)[0],
        "entangled_feature_map": FEATURE_MAPS["entangled"](n_features)[0],
        "zz_feature_map": FEATURE_MAPS["zz"](n_features)[0],
        "vqc_ansatz": build_vqc_circuit(n_features, n_layers=2)[0],
        "data_reuploading": reuploading_layer(n_features, n_layers=3)[0],
        "qcnn": build_qcnn_circuit(n_qcnn_features)[0],
    }
    for name, circ in circuits.items():
        fig = circ.draw("mpl", fold=-1)
        fig.savefig(PLOTS_DIR / "circuits" / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    logger.info("Saved %d circuit diagrams to %s", len(circuits), PLOTS_DIR / "circuits")


# --- Stage 3: quantum feature map + models -----------------------------------

def stage_quantum(data: pd.DataFrame, selection: dict, execution_config: ExecutionConfig, max_samples: int | None) -> dict:
    from sklearn.preprocessing import MinMaxScaler

    selected_features = selection["selected_features"]
    X = data[selected_features].values
    y = data[CLASSIFICATION_TARGET].values

    if max_samples is not None and max_samples < len(X):
        logger.warning(
            "max_samples=%d < %d: running a REDUCED, non-full LOOCV for practicality on backend=%s. "
            "Results are a smoke test, not the Task 4 deliverable.",
            max_samples, len(X), execution_config.label(),
        )
        X, y = X[:max_samples], y[:max_samples]

    plot_circuit_diagrams(n_features=len(selected_features), n_qcnn_features=len(selection["features_by_k"][6]))

    # Task 3: kernel heatmap + KTA, per feature map -- one shared executor.
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

    # Task 4: full quantum model suite under LOOCV. The QCNN is architecturally
    # fixed at 6 qubits (per the challenge spec), so it runs on the dedicated
    # 6-feature subset rather than whichever k the automated selector found
    # optimal for the other models.
    non_qcnn_names = [n for n in QUANTUM_MODEL_BUILDERS if n != "QCNN"]
    quantum_results = run_quantum_classification_suite(X, y, execution_config=execution_config, model_names=non_qcnn_names)

    X_qcnn = data[selection["features_by_k"][6]].values
    if max_samples is not None and max_samples < len(X_qcnn):
        X_qcnn = X_qcnn[:max_samples]
    logger.info("Running QCNN on its dedicated 6-feature subset: %s", selection["features_by_k"][6])
    quantum_results.update(
        run_quantum_classification_suite(X_qcnn, y, execution_config=execution_config, model_names=["QCNN"])
    )

    q_table = summarize_results(quantum_results, kind="classification")
    save_results_table(q_table, "quantum_loocv.csv")
    logger.info("Quantum LOOCV results (backend=%s):\n%s", execution_config.label(), q_table.to_string())

    # Per-drug score vs. true COMDR_15min scatter (best-accuracy quantum model).
    best_name = q_table.iloc[0]["model"]
    best_res = quantum_results[best_name]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(data[REGRESSION_TARGET].values[: len(best_res["y_true"])], best_res["y_proba"],
               c=best_res["y_true"], cmap="coolwarm")
    ax.set_xlabel("True COMDR_15min")
    ax.set_ylabel(f"P(|1>) -- {best_name}")
    ax.set_title(f"Per-drug quantum score vs. true COMDR_15min ({best_name})")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "scatter" / "score_vs_comdr.png", dpi=150)
    plt.close(fig)

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
    kta_model = KTAOptimizedQuantumKernel(executor=kta_executor, n_layers=1, maxiter=40)
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
    ax.set_title("Classical (blue) vs. Quantum (orange) -- unified LOOCV comparison")
    plt.xticks(rotation=30, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "unified_comparison.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="OQI Hackathon 2026 QML pipeline")
    parser.add_argument(
        "--stage", choices=["fetch", "features", "classical", "quantum", "bonus", "all"], default="all",
    )
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

    if args.stage in ("classical", "all"):
        clf_results = stage_classical(data, selection)
    if args.stage in ("quantum", "all"):
        quantum_results = stage_quantum(data, selection, execution_config, args.max_samples)
    if args.stage in ("bonus", "all"):
        stage_bonus(data, selection, execution_config, args.max_samples)
    if args.stage == "all" and clf_results is not None and quantum_results is not None:
        stage_unified_comparison(clf_results, quantum_results)

    logger.info("Pipeline stage '%s' complete (backend=%s).", args.stage, execution_config.label())


if __name__ == "__main__":
    main()
