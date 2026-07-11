"""Task 5 deliverable: unified classical-vs-quantum comparison plots.

Reads one normalized JSON per model from a results directory (written by
`src.evaluation.save_results_json`, called automatically by every
`main.py` classification/regression stage) and renders six figures:

    1. classification_comparison   -- F1 / recall_1 / accuracy, grouped bars
    2. regression_comparison       -- Q^2 / RMSE vs the Patzmann benchmark
    3. per_drug_regression_scatter -- predicted vs. true COMDR15, LOOCV
    4. per_drug_classification_heatmap -- correct/wrong per model per drug
    5. confusion_matrices_classical -- one 2x2 confusion matrix per classical model
    6. confusion_matrices_quantum   -- one 2x2 confusion matrix per quantum model

Every figure marks classical vs. quantum unambiguously through at least
three redundant channels at once (never just one): a solid colored zone
band with a bold "CLASSICAL ML" / "QUANTUM ML" header, a thick divider
line between the two groups, and paradigm-colored axis tick labels. A
"Δ" callout box states the classical-vs-quantum gap as a single signed
number, so the comparison never depends on eyeballing bar heights alone.

Each JSON has the shape:
    {
        "metrics": {...},        # accuracy/f1/recall_0/recall_1/elapsed_s
                                  # or r2/q2/mae/rmse/elapsed_s
        "predictions": [...],    # one entry per drug, LOOCV order
        "targets": [...],
        "scores": [...],         # P(|1>) for classifiers, predictions for regressors
    }

Usage:
    python scripts/plot_model_comparison.py --results_dir data/results
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from sklearn.metrics import confusion_matrix

# --- Project model registry ---------------------------------------------
# Kept in sync by hand with QUANTUM_MODEL_BUILDERS / QUANTUM_REGRESSION_MODEL_BUILDERS
# (src/quantum_models.py) and CLASSICAL_CLASSIFIERS (src/classical_models.py).
# A model listed here with no matching JSON in --results_dir is skipped
# with a warning rather than raising, so the script still produces partial
# plots while a suite is only partially run.
CLASSIFICATION_CLASSICAL_MODELS = ["SVC_rbf", "RandomForest", "GradientBoosting"]
CLASSIFICATION_QUANTUM_MODELS = ["QK-SVM_angle", "QK-SVM_trained", "VQC", "QCNN"]
REGRESSION_CLASSICAL_MODELS = ["PLS_regression"]
REGRESSION_QUANTUM_MODELS = ["QK-KRR_angle", "VQR_spsa_cobyla", "QCNN-R"]

PATZMANN_Q2_BENCHMARK = 0.77
CLASSIFICATION_THRESHOLD = 2.0  # COMDR15 > 2.0 => Responder (challenge brief, Eq. 2)

# One unambiguous color pair, used everywhere a model's paradigm needs to
# read out at a glance: zone bands, header text, tick labels, panel
# borders, legend swatches. Never used for anything else, so "blue" and
# "orange/red" mean exactly one thing in every figure this script makes.
COLOR_CLASSICAL = "#2E5FA3"   # strong blue: classical ML
COLOR_QUANTUM = "#D2691E"     # strong burnt orange: quantum ML
COLOR_PATZMANN = "black"

TITLE_FONTSIZE = 15
LABEL_FONTSIZE = 12
TICK_FONTSIZE = 10
HEADER_FONTSIZE = 12
DPI = 300


def load_results(results_dir: str, model_names: list[str] | None = None) -> dict[str, dict]:
    """Load one normalized result dict per model from `results_dir`.

    Reads `<results_dir>/<model_name>.json` for each name in
    `model_names` (or every `*.json` file present, if `model_names` is
    None). Missing files are skipped with a printed warning instead of
    raising, so a partially-run results directory still produces whatever
    plots its available models support. For testing, skip this function
    entirely and pass an in-memory `{model_name: result_dict}` dict
    straight to the `plot_*` functions below -- they never touch disk
    themselves.

    Parameters
    ----------
    results_dir : str
        Directory containing one `<model_name>.json` file per model.
    model_names : list[str], optional
        Restrict loading to these models. Defaults to every `*.json` file
        found in `results_dir`.

    Returns
    -------
    dict[str, dict]
        {model_name: {"metrics": {...}, "predictions": [...], "targets": [...], "scores": [...]}}
    """
    results_path = Path(results_dir)
    names = model_names if model_names is not None else [p.stem for p in results_path.glob("*.json")]

    results = {}
    for name in names:
        file_path = results_path / f"{name}.json"
        if not file_path.exists():
            print(f"[load_results] WARNING: {file_path} not found, skipping '{name}'")
            continue
        with open(file_path) as f:
            results[name] = json.load(f)
    return results


def _save_fig(fig: plt.Figure, output_dir: Path, basename: str) -> None:
    """Save `fig` as both `<basename>.png` (300 dpi) and `<basename>.pdf`
    under `output_dir`, then close it.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{basename}.png"
    pdf_path = output_dir / f"{basename}.pdf"
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {png_path} and {pdf_path}")


def _mark_paradigm_zones(ax: plt.Axes, n_classical: int, n_total: int) -> None:
    """Three redundant, unmistakable markers of the classical/quantum split
    along the x-axis: a solid colored background band per zone, a bold
    "CLASSICAL ML" / "QUANTUM ML" header text pinned to the top of each
    band, and a thick black divider line at the boundary between them.
    Deliberately stronger than a subtle tint -- this needs to read
    correctly even in a quick glance or a small thumbnail.
    """
    if n_classical > 0:
        ax.axvspan(-0.5, n_classical - 0.5, color=COLOR_CLASSICAL, alpha=0.14, zorder=0)
        ax.text(
            (n_classical - 1) / 2, 1.02, "CLASSICAL ML", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=HEADER_FONTSIZE, fontweight="bold", color=COLOR_CLASSICAL,
        )
    if n_total > n_classical:
        ax.axvspan(n_classical - 0.5, n_total - 0.5, color=COLOR_QUANTUM, alpha=0.14, zorder=0)
        ax.text(
            n_classical + (n_total - n_classical - 1) / 2, 1.02, "QUANTUM ML", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=HEADER_FONTSIZE, fontweight="bold", color=COLOR_QUANTUM,
        )
    if 0 < n_classical < n_total:
        ax.axvline(n_classical - 0.5, color="black", linewidth=2.4, zorder=5)


def _color_xtick_labels(ax: plt.Axes, models: list[str], classical_models: set[str]) -> None:
    """Color each x-tick label to match its model's paradigm (blue for
    classical, orange for quantum) -- a second, independent way to tell
    the groups apart that survives even if the figure is printed in a
    context where the background shading is hard to see.
    """
    for label, name in zip(ax.get_xticklabels(), models):
        label.set_color(COLOR_CLASSICAL if name in classical_models else COLOR_QUANTUM)
        label.set_fontweight("bold")


def _delta_callout(ax: plt.Axes, best_classical: float, best_quantum: float, metric_name: str, fmt: str = "{:+.3f}") -> None:
    """A single boxed, signed-number annotation stating the gap between
    the best classical and best quantum result for one metric, so the
    comparison this whole figure is making does not depend on visually
    comparing bar heights -- it is also just stated as one unambiguous
    number, "quantum leads/trails by X".
    """
    delta = best_quantum - best_classical
    verdict = "QUANTUM LEADS" if delta > 0 else ("CLASSICAL LEADS" if delta < 0 else "TIE")
    color = COLOR_QUANTUM if delta > 0 else (COLOR_CLASSICAL if delta < 0 else "grey")
    text = f"Δ best {metric_name} = {fmt.format(delta)}  →  {verdict}"
    ax.text(
        0.5, -0.22, text, transform=ax.transAxes, ha="center", va="top",
        fontsize=LABEL_FONTSIZE, fontweight="bold", color="white",
        bbox=dict(facecolor=color, alpha=0.92, edgecolor="none", boxstyle="round,pad=0.4"),
    )


def plot_classification_comparison(
    results: dict[str, dict],
    classical_models: list[str],
    quantum_models: list[str],
    output_dir: Path,
) -> None:
    """Grouped bar chart: F1 (primary), Responder recall / recall_1
    (secondary), and accuracy (tertiary, reference only) for every
    classification model, classical and quantum, each group internally
    sorted by F1 descending.

    What to look for: the blue "CLASSICAL ML" zone vs. the orange
    "QUANTUM ML" zone, separated by a thick divider -- and the boxed Δ
    callout under the x-axis states the winner in one number. Within
    that, bars crossing the "Classical baseline" dashed line are quantum
    models beating the best classical F1. Watch for a model whose
    recall_1 bar sits well below its F1 bar -- that model is
    inconsistently catching true Responders despite a decent overall F1,
    which in this pharma context is the costliest kind of error (a missed
    formulation opportunity), not a cosmetic one.
    """
    def f1_of(name: str) -> float:
        return results[name]["metrics"]["f1"]

    classical_sorted = sorted((m for m in classical_models if m in results), key=f1_of, reverse=True)
    quantum_sorted = sorted((m for m in quantum_models if m in results), key=f1_of, reverse=True)
    models = classical_sorted + quantum_sorted
    if not models:
        print("[plot_classification_comparison] no models found in results, skipping")
        return

    f1 = [results[m]["metrics"]["f1"] for m in models]
    recall1 = [results[m]["metrics"]["recall_1"] for m in models]
    acc = [results[m]["metrics"]["accuracy"] for m in models]

    x = np.arange(len(models))
    width = 0.26
    fig, ax = plt.subplots(figsize=(max(11.0, len(models) * 1.4), 6.6))

    _mark_paradigm_zones(ax, len(classical_sorted), len(models))

    ax.bar(x - width, f1, width, label="F1 score (primary)", color="#1B3B6F", zorder=3, edgecolor="black", linewidth=0.4)
    ax.bar(x, recall1, width, label="Recall — Responder (recall_1)", color="#E67E22", zorder=3, edgecolor="black", linewidth=0.4)
    ax.bar(x + width, acc, width, label="Accuracy (reference only)", color="#BDBDBD", zorder=3, edgecolor="black", linewidth=0.4)

    if classical_sorted:
        best_classical_f1 = f1_of(classical_sorted[0])
        ax.axhline(best_classical_f1, color=COLOR_PATZMANN, linestyle="--", linewidth=1.3, zorder=4)
        ax.text(
            -0.4, best_classical_f1 + 0.015, f"Classical baseline (F1={best_classical_f1:.3f})",
            ha="left", va="bottom", fontsize=TICK_FONTSIZE, style="italic", zorder=5,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.5),
        )

    # Annotate every quantum model's F1 directly on its bar -- with only 4
    # quantum models in this suite, "top-3" would hide barely anything, so
    # every one gets its value labeled to remove all ambiguity.
    for name in quantum_sorted:
        idx = models.index(name)
        val = f1_of(name)
        ax.text(
            idx - width, val + 0.015, f"{val:.3f}", ha="center", va="bottom",
            fontsize=TICK_FONTSIZE, fontweight="bold", color="#1B3B6F",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right", fontsize=TICK_FONTSIZE)
    _color_xtick_labels(ax, models, set(classical_sorted))
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=LABEL_FONTSIZE)
    ax.set_title("LOOCV Classification Results — Classical vs Quantum (N=29)", fontsize=TITLE_FONTSIZE, fontweight="bold", pad=28)
    ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.legend(fontsize=TICK_FONTSIZE, loc="upper right", framealpha=0.9)

    if classical_sorted and quantum_sorted:
        _delta_callout(ax, f1_of(classical_sorted[0]), f1_of(quantum_sorted[0]), "F1")

    fig.tight_layout()
    _save_fig(fig, output_dir, "classification_comparison")


def plot_regression_comparison(
    results: dict[str, dict],
    classical_models: list[str],
    quantum_models: list[str],
    output_dir: Path,
) -> None:
    """Grouped bar chart: Q^2 (LOOCV, primary, left axis, green) and RMSE
    (secondary, right axis, red, COMDR15 units) for every regression
    model, sorted by Q^2 descending.

    What to look for: the blue "CLASSICAL ML" zone vs. the orange
    "QUANTUM ML" zone, separated by a thick divider -- and the boxed Δ
    callout under the x-axis states the winner in one number. Within
    that, the Pätzmann Q^2=0.77 dashed line is the single number every
    bar is implicitly judged against -- a bar crossing it matches or
    beats the published classical chemometric benchmark on this exact
    dataset. The Q^2=0 line matters just as much: a bar sitting at or
    below it means that model does no better (or worse) than always
    predicting the training-fold mean, no matter how small its RMSE looks
    in isolation.
    """
    def q2_of(name: str) -> float:
        return results[name]["metrics"]["q2"]

    classical_sorted = sorted((m for m in classical_models if m in results), key=q2_of, reverse=True)
    quantum_sorted = sorted((m for m in quantum_models if m in results), key=q2_of, reverse=True)
    models = classical_sorted + quantum_sorted
    if not models:
        print("[plot_regression_comparison] no models found in results, skipping")
        return

    q2 = [results[m]["metrics"]["q2"] for m in models]
    rmse = [results[m]["metrics"]["rmse"] for m in models]

    x = np.arange(len(models))
    width = 0.4
    fig, ax1 = plt.subplots(figsize=(max(9.0, len(models) * 1.9), 6.6))

    _mark_paradigm_zones(ax1, len(classical_sorted), len(models))

    ax1.bar(x - width / 2, q2, width, label="Q² (LOOCV)", color="#1B5E20", zorder=3, edgecolor="black", linewidth=0.4)
    for xi, val in zip(x, q2):
        ax1.text(
            xi - width / 2, val + (0.02 if val >= 0 else -0.05), f"{val:.3f}",
            ha="center", va="bottom" if val >= 0 else "top", fontsize=TICK_FONTSIZE, fontweight="bold",
        )

    # Both reference labels anchored at the left edge (not right), so
    # neither collides with the "upper right" legend. Pushed further above
    # the line (+0.07, not +0.02) and given a white backing box, since a
    # model landing close to the Patzmann Q^2 (as PLS itself does, by
    # construction) would otherwise have its own bar-top value annotation
    # sitting right where this label would go.
    ax1.axhline(PATZMANN_Q2_BENCHMARK, color=COLOR_PATZMANN, linestyle="--", linewidth=1.3, zorder=4)
    ax1.text(
        -0.45, PATZMANN_Q2_BENCHMARK + 0.07,
        f"Pätzmann benchmark (PLS, Q²={PATZMANN_Q2_BENCHMARK:.2f})",
        ha="left", va="bottom", fontsize=TICK_FONTSIZE, style="italic", zorder=5,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.5),
    )
    ax1.axhline(0.0, color="grey", linestyle=":", linewidth=1.1, zorder=4)
    ax1.text(
        -0.45, 0.02, "Baseline (predict mean)",
        ha="left", va="bottom", fontsize=TICK_FONTSIZE - 1, color="grey", style="italic", zorder=5,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.5),
    )

    ax1.set_ylabel("Q² (LOOCV)", fontsize=LABEL_FONTSIZE, color="#1B5E20")
    ax1.set_ylim(min(-0.5, min(q2) - 0.1), 1.05)
    ax1.tick_params(axis="y", labelcolor="#1B5E20", labelsize=TICK_FONTSIZE)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=20, ha="right", fontsize=TICK_FONTSIZE)
    _color_xtick_labels(ax1, models, set(classical_sorted))

    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, rmse, width, label="RMSE (COMDR15 units)", color="#C0392B", alpha=0.85, zorder=3, edgecolor="black", linewidth=0.4)
    ax2.set_ylabel("RMSE (COMDR15 units)", fontsize=LABEL_FONTSIZE, color="#C0392B")
    ax2.tick_params(axis="y", labelcolor="#C0392B", labelsize=TICK_FONTSIZE)

    ax1.set_title("LOOCV Regression Results — Q² vs Pätzmann Benchmark (N=29)", fontsize=TITLE_FONTSIZE, fontweight="bold", pad=28)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=TICK_FONTSIZE, loc="upper right", framealpha=0.9)

    if classical_sorted and quantum_sorted:
        _delta_callout(ax1, q2_of(classical_sorted[0]), q2_of(quantum_sorted[0]), "Q²")

    fig.tight_layout()
    _save_fig(fig, output_dir, "regression_comparison")


def plot_per_drug_regression_scatter(
    results: dict[str, dict],
    best_quantum_model: str,
    classical_model: str,
    drug_names: list[str] | None,
    output_dir: Path,
) -> None:
    """Two-panel scatter of predicted vs. true COMDR15 (one point per
    LOOCV-held-out drug) for the Patzmann PLS baseline and the best
    quantum regressor.

    What to look for: each panel has a colored border and header matching
    its paradigm (blue classical, orange quantum) and its own Q²/RMSE in
    the title, plus a boxed Δ callout across the top stating the Q² gap
    between them as one signed number. Within each panel, points on the
    black y=x diagonal are perfect LOOCV predictions; distance from it is
    per-drug error. Point color follows the *classification* threshold
    (COMDR15=2.0) even in this regression view: green means the model
    landed on the correct side of the Responder/Non-Responder boundary
    regardless of its exact numeric error, red means it crossed to the
    wrong side. Labeled points are drugs with |predicted - true| > 1.5,
    worth cross-referencing against physicochemical outliers.
    """
    candidates = [(classical_model, "CLASSICAL", COLOR_CLASSICAL), (best_quantum_model, "QUANTUM", COLOR_QUANTUM)]
    models_to_plot = [(m, label, color) for m, label, color in candidates if m in results]
    if not models_to_plot:
        print("[plot_per_drug_regression_scatter] no models found in results, skipping")
        return

    fig, axes = plt.subplots(1, len(models_to_plot), figsize=(7.4 * len(models_to_plot), 7.0), squeeze=False)
    axes = axes[0]

    for ax, (model_name, label, color) in zip(axes, models_to_plot):
        res = results[model_name]
        y_true = np.asarray(res["targets"], dtype=float)
        y_pred = np.asarray(res["predictions"], dtype=float)
        m = res["metrics"]

        correct_side = (y_true > CLASSIFICATION_THRESHOLD) == (y_pred > CLASSIFICATION_THRESHOLD)
        colors = np.where(correct_side, "#2E7D32", "#C0392B")

        ax.scatter(y_true, y_pred, c=colors, s=65, edgecolor="k", linewidth=0.6, zorder=3)

        axis_max = float(max(y_true.max(), y_pred.max())) * 1.08
        lims = [0.0, axis_max]
        ax.plot(lims, lims, color=COLOR_PATZMANN, linestyle="--", linewidth=1.1, zorder=2, label="y = x (perfect prediction)")
        ax.axvline(CLASSIFICATION_THRESHOLD, color="grey", linestyle=":", linewidth=1)
        ax.axhline(CLASSIFICATION_THRESHOLD, color="grey", linestyle=":", linewidth=1)

        errors = np.abs(y_pred - y_true)
        for i in np.where(errors > 1.5)[0]:
            label_text = drug_names[i] if drug_names is not None and i < len(drug_names) else str(i)
            ax.annotate(
                label_text, (y_true[i], y_pred[i]), fontsize=TICK_FONTSIZE - 1,
                xytext=(5, 5), textcoords="offset points",
            )

        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("True COMDR15", fontsize=LABEL_FONTSIZE)
        ax.set_ylabel("Predicted COMDR15", fontsize=LABEL_FONTSIZE)
        # Colored header banner names the paradigm unambiguously above each panel.
        ax.text(
            0.5, 1.14, label, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=HEADER_FONTSIZE + 1, fontweight="bold", color="white",
            bbox=dict(facecolor=color, alpha=0.95, edgecolor="none", boxstyle="round,pad=0.35"),
        )
        ax.set_title(f"{model_name}\nQ²={m['q2']:.3f}, RMSE={m['rmse']:.3f}", fontsize=TITLE_FONTSIZE - 1, pad=14)
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.2)
        ax.tick_params(labelsize=TICK_FONTSIZE)
        ax.legend(fontsize=TICK_FONTSIZE - 1, loc="lower right")

    if len(models_to_plot) == 2:
        q2_a = results[models_to_plot[0][0]]["metrics"]["q2"]
        q2_b = results[models_to_plot[1][0]]["metrics"]["q2"]
        delta = q2_b - q2_a  # quantum - classical, by construction of `candidates` order
        verdict = "QUANTUM LEADS" if delta > 0 else ("CLASSICAL LEADS" if delta < 0 else "TIE")
        color = COLOR_QUANTUM if delta > 0 else (COLOR_CLASSICAL if delta < 0 else "grey")
        fig.text(
            0.5, 0.965, f"Δ Q² (quantum − classical) = {delta:+.3f}  →  {verdict}",
            ha="center", va="top", fontsize=LABEL_FONTSIZE, fontweight="bold", color="white",
            bbox=dict(facecolor=color, alpha=0.92, edgecolor="none", boxstyle="round,pad=0.4"),
        )

    fig.suptitle(
        "Per-Drug LOOCV Predictions — Best Quantum Regressor vs PLS Baseline",
        fontsize=TITLE_FONTSIZE, fontweight="bold", y=1.06,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    _save_fig(fig, output_dir, "per_drug_regression_scatter")


def plot_per_drug_classification_heatmap(
    results: dict[str, dict],
    drug_names: list[str],
    true_comdr15: np.ndarray,
    output_dir: Path,
    classical_models: list[str] | None = None,
) -> None:
    """(n_models x 29 drugs) heatmap: green if that model's LOOCV
    prediction for that drug was correct, red if wrong. Rows sorted by F1
    (best model at top), columns sorted by true COMDR15 ascending, with a
    vertical divider at the Responder/Non-Responder boundary.

    What to look for: each model's row label is colored and bolded by
    paradigm (blue classical, orange quantum), so paradigm reads out
    unambiguously even though rows are sorted by rank (best model first)
    rather than grouped -- ranking determines position, color alone
    determines paradigm. A column that is mostly red straight down, across
    every row regardless of family, identifies a genuinely hard drug --
    most likely one sitting close to the COMDR15=2.0 threshold or an
    outlier in feature space -- rather than a weakness specific to any one
    model. That is the direct visual answer to "which drugs are hardest to
    predict across all models" (Task 5).
    """
    classical_set = set(classical_models or CLASSIFICATION_CLASSICAL_MODELS)
    models = sorted(results.keys(), key=lambda m: -results[m]["metrics"]["f1"])
    if not models:
        print("[plot_per_drug_classification_heatmap] no models found in results, skipping")
        return

    order = np.argsort(true_comdr15)
    sorted_drugs = [drug_names[i] for i in order]
    sorted_comdr = np.asarray(true_comdr15)[order]

    correctness = np.zeros((len(models), len(order)))
    for row, name in enumerate(models):
        y_true = np.asarray(results[name]["targets"])
        y_pred = np.asarray(results[name]["predictions"])
        correct = (y_true == y_pred).astype(int)
        correctness[row] = correct[order]

    fig, ax = plt.subplots(figsize=(max(14.0, len(order) * 0.55), max(5.5, len(models) * 0.7)))
    cmap = ListedColormap(["#C0392B", "#2E7D32"])  # 0 -> wrong (red), 1 -> correct (green)
    sns.heatmap(
        correctness, cmap=cmap, cbar=False, linewidths=0.6, linecolor="white",
        xticklabels=[f"{d}\n{c:.2f}" for d, c in zip(sorted_drugs, sorted_comdr)],
        yticklabels=models, ax=ax, vmin=0, vmax=1,
    )

    # Responder/Non-Responder header, in AXES fraction (not data) coordinates
    # so it sits reliably just above the heatmap regardless of row count,
    # instead of colliding with the title.
    responder_boundary = int(np.searchsorted(sorted_comdr, CLASSIFICATION_THRESHOLD))
    ax.axvline(responder_boundary, color="black", linewidth=2.4)
    header_y = 1.03
    if responder_boundary > 0:
        ax.text(
            responder_boundary / 2, header_y, "Non-Responder", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=TICK_FONTSIZE, fontweight="bold",
        )
    if responder_boundary < len(order):
        ax.text(
            responder_boundary + (len(order) - responder_boundary) / 2, header_y, "Responder",
            transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=TICK_FONTSIZE, fontweight="bold",
        )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center", fontsize=TICK_FONTSIZE - 2)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=TICK_FONTSIZE, fontweight="bold")
    for label, name in zip(ax.get_yticklabels(), models):
        label.set_color(COLOR_CLASSICAL if name in classical_set else COLOR_QUANTUM)

    # Legend placed to the RIGHT of the heatmap, vertically centered --
    # the x-tick zone below the heatmap is already tall (two-line, 90-
    # degree-rotated drug labels), so anything anchored below it would
    # need an awkwardly large negative offset to clear it reliably.
    legend_handles = [
        Patch(facecolor=COLOR_CLASSICAL, label="Classical ML"),
        Patch(facecolor=COLOR_QUANTUM, label="Quantum ML"),
        Patch(facecolor="#2E7D32", label="Correct prediction"),
        Patch(facecolor="#C0392B", label="Wrong prediction"),
    ]
    ax.legend(
        handles=legend_handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
        fontsize=TICK_FONTSIZE, frameon=False,
    )

    ax.set_title(
        "Per-Drug Prediction Correctness Across All Classification Models",
        fontsize=TITLE_FONTSIZE, fontweight="bold", pad=34,
    )
    fig.tight_layout()
    _save_fig(fig, output_dir, "per_drug_classification_heatmap")


def plot_confusion_matrices(
    results: dict[str, dict],
    model_names: list[str],
    output_dir: Path,
    suite_label: str,
    color: str,
    filename: str,
) -> None:
    """One figure holding a 2x2 confusion matrix per model in
    `model_names`, laid out side by side in a single row. Rows = true
    label, columns = predicted label (Non-Responder=0, Responder=1), the
    standard sklearn convention. Every subplot is colored in a single-hue
    colormap derived from `color`, and bordered in that same color, so
    the whole figure (and each subplot within it) is unmistakably tagged
    to one paradigm even cropped out of context -- this is meant to be
    called once for the classical models and once for the quantum models,
    producing two separate, self-labeled images.

    What to look for: the diagonal (top-left TN, bottom-right TP) holds
    correct predictions -- a model that "works" has essentially all its
    mass on the diagonal. Off-diagonal mass is where it matters *which*
    cell: top-right (FP) is a false alarm (predicted Responder, actually
    isn't), bottom-left (FN) is a missed Responder -- the costliest error
    in this pharma context, a genuine formulation opportunity the model
    told you to skip.
    """
    present = [m for m in model_names if m in results]
    if not present:
        print(f"[plot_confusion_matrices] no models found for '{suite_label}', skipping")
        return

    ncols = min(len(present), 4)
    nrows = -(-len(present) // ncols)  # ceiling division
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 5.0 * nrows), squeeze=False)
    axes_flat = axes.flatten()
    cmap = sns.light_palette(color, as_cmap=True)

    for ax, name in zip(axes_flat, present):
        res = results[name]
        y_true = np.asarray(res["targets"], dtype=int)
        y_pred = np.asarray(res["predictions"], dtype=int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        m = res["metrics"]

        sns.heatmap(
            cm, annot=True, fmt="d", cmap=cmap, cbar=False, ax=ax, square=True,
            annot_kws={"fontsize": 18, "fontweight": "bold"}, linewidths=1.5, linecolor="white",
            xticklabels=["Non-Resp.", "Responder"], yticklabels=["Non-Resp.", "Responder"],
        )
        ax.set_xlabel("Predicted", fontsize=LABEL_FONTSIZE)
        ax.set_ylabel("True", fontsize=LABEL_FONTSIZE)
        ax.set_title(
            f"{name}\nAcc={m['accuracy']:.3f}  F1={m['f1']:.3f}",
            fontsize=TITLE_FONTSIZE - 2, fontweight="bold", color=color,
        )
        ax.tick_params(labelsize=TICK_FONTSIZE - 1, rotation=0)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(color)
            spine.set_linewidth(2.2)

    for ax in axes_flat[len(present):]:
        ax.axis("off")

    fig.suptitle(
        f"Confusion Matrices — {suite_label} (LOOCV, N=29)",
        fontsize=TITLE_FONTSIZE + 2, fontweight="bold", color=color, y=1.04,
    )
    fig.tight_layout()
    _save_fig(fig, output_dir, filename)


def main() -> None:
    from src.config import MODELING_TABLE_CSV, PLOTS_DIR, RESULTS_DIR

    parser = argparse.ArgumentParser(description="Build the unified classical-vs-quantum comparison plots (Task 5).")
    parser.add_argument(
        "--results_dir", type=str, default=str(RESULTS_DIR),
        help="Directory containing one <model_name>.json per model (default: data/results/).",
    )
    parser.add_argument(
        "--output_dir", type=str, default=str(PLOTS_DIR / "comparison"),
        help="Directory to save the generated plots (default: plots/comparison/).",
    )
    parser.add_argument(
        "--data_csv", type=str, default=str(MODELING_TABLE_CSV),
        help="CSV with 'drug' and 'COMDR_15min' columns, in the same row order as the LOOCV runs "
             "(default: data/processed/modeling_table.csv). Used for Plot 3's outlier labels and all of Plot 4.",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    clf_results = load_results(args.results_dir, CLASSIFICATION_CLASSICAL_MODELS + CLASSIFICATION_QUANTUM_MODELS)
    reg_results = load_results(args.results_dir, REGRESSION_CLASSICAL_MODELS + REGRESSION_QUANTUM_MODELS)

    plot_classification_comparison(clf_results, CLASSIFICATION_CLASSICAL_MODELS, CLASSIFICATION_QUANTUM_MODELS, output_dir)
    plot_regression_comparison(reg_results, REGRESSION_CLASSICAL_MODELS, REGRESSION_QUANTUM_MODELS, output_dir)
    plot_confusion_matrices(
        clf_results, CLASSIFICATION_CLASSICAL_MODELS, output_dir, "Classical ML", COLOR_CLASSICAL, "confusion_matrices_classical"
    )
    plot_confusion_matrices(
        clf_results, CLASSIFICATION_QUANTUM_MODELS, output_dir, "Quantum ML", COLOR_QUANTUM, "confusion_matrices_quantum"
    )

    drug_names, true_comdr15 = None, None
    try:
        data = pd.read_csv(args.data_csv)
        drug_names = data["drug"].tolist()
        true_comdr15 = data["COMDR_15min"].values
    except (FileNotFoundError, KeyError) as e:
        print(f"[main] could not load drug metadata from {args.data_csv} ({e}); Plot 3 will use numeric drug "
              f"indices and Plot 4 will be skipped.")

    quantum_reg_present = [m for m in REGRESSION_QUANTUM_MODELS if m in reg_results]
    classical_reg_present = [m for m in REGRESSION_CLASSICAL_MODELS if m in reg_results]
    if quantum_reg_present and classical_reg_present:
        best_quantum_reg = max(quantum_reg_present, key=lambda m: reg_results[m]["metrics"]["q2"])
        plot_per_drug_regression_scatter(reg_results, best_quantum_reg, classical_reg_present[0], drug_names, output_dir)
    else:
        print("[main] missing a classical or quantum regressor result, skipping Plot 3.")

    if true_comdr15 is not None:
        plot_per_drug_classification_heatmap(
            clf_results, drug_names, true_comdr15, output_dir, classical_models=CLASSIFICATION_CLASSICAL_MODELS
        )


if __name__ == "__main__":
    main()
