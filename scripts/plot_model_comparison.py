"""Task 5 deliverable: unified classical-vs-quantum comparison plots.

Reads one normalized JSON per model from a results directory (written by
`src.evaluation.save_results_json`, called automatically by every
`main.py` classification/regression stage) and renders eight figures:

    1. classification_comparison       -- F1 / recall_1 / accuracy, grouped bars
    2. roc_comparison                  -- ROC curves + AUC, every classifier on one axis
    3. confusion_matrices_classical    -- one 2x2 confusion matrix per classical model
    4. confusion_matrices_quantum      -- one 2x2 confusion matrix per quantum model
    5. regression_comparison           -- Q^2 vs RMSE, two panels (never one dual-axis plot)
    6. per_drug_regression_scatter     -- predicted vs. true COMDR15, LOOCV
    7. per_drug_classification_heatmap -- correct/wrong per model per drug
    8. kta_comparison                  -- fixed feature maps vs. the trained kernel's
                                           KTA before/after optimization (needs the
                                           kernel_kta_by_feature_map.csv / kta_optimization.csv
                                           that main.py's quantum-classification / bonus
                                           stages write into --results_dir)

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

`main.py --stage all` also calls this module's `main()` automatically at
the end of a full run, so these figures are produced without a separate
manual step in the common case; run this script by hand only to rebuild
plots from an already-populated --results_dir without rerunning LOOCV.
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
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

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

# --- Palette -------------------------------------------------------------
# A validated categorical palette (fixed order = the CVD-safety mechanism --
# never cycled, never reassigned by rank/filter). Slot 1 (blue) and slot 8
# (orange) double as the two-family "classical vs quantum" identity used for
# zone shading, the accuracy/Q^2 bars, and the confusion-matrix borders;
# the full 8-slot order is used wherever every individual model needs its
# own identity (the ROC comparison).
PALETTE_CATEGORICAL = [
    "#2a78d6",  # 1 blue
    "#1baf7a",  # 2 aqua
    "#eda100",  # 3 yellow
    "#008300",  # 4 green
    "#4a3aa7",  # 5 violet
    "#e34948",  # 6 red
    "#e87ba4",  # 7 magenta
    "#eb6834",  # 8 orange
]
COLOR_CLASSICAL_ZONE = PALETTE_CATEGORICAL[0]  # blue: classical-model identity
COLOR_QUANTUM_ZONE = PALETTE_CATEGORICAL[7]    # orange: quantum-model identity
COLOR_PATZMANN = "#0b0b0b"       # primary ink -- reference/benchmark lines
COLOR_MUTED = "#898781"          # muted ink -- chance lines, secondary annotations
COLOR_GRID = "#e1e0d9"           # hairline gridline, one shade off the surface

TITLE_FONTSIZE = 14
LABEL_FONTSIZE = 12
TICK_FONTSIZE = 10
DPI = 300


def _style_axis(ax: plt.Axes, y_grid: bool = True) -> None:
    """Shared professional-chart chrome: hairline hyaline gridlines instead
    of matplotlib's default heavy border, no top/right spine (nothing to
    frame), and muted tick labels -- applied to every plot in this module
    so the figures read as one consistent, deliberately designed set
    rather than independently-styled ad hoc charts.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLOR_GRID)
        ax.spines[side].set_linewidth(0.9)
    if y_grid:
        ax.yaxis.grid(True, color=COLOR_GRID, linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=COLOR_MUTED, labelsize=TICK_FONTSIZE)


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


def _shade_group_zones(ax: plt.Axes, n_classical: int, n_total: int) -> None:
    """Faint background shading marking the classical-model x-range (blue)
    vs. the quantum-model x-range (orange), so the two groups are visually
    separated without needing a second color scheme for the bars themselves.
    """
    if n_classical > 0:
        ax.axvspan(-0.5, n_classical - 0.5, color=COLOR_CLASSICAL_ZONE, alpha=0.06, zorder=0)
    if n_total > n_classical:
        ax.axvspan(n_classical - 0.5, n_total - 0.5, color=COLOR_QUANTUM_ZONE, alpha=0.06, zorder=0)


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

    What to look for: bars crossing the "Classical baseline" dashed line
    are quantum models beating the best classical F1. More importantly,
    watch for a model whose recall_1 bar sits well below its F1 bar --
    that model is inconsistently catching true Responders despite a
    decent overall F1, which in this pharma context is the costliest kind
    of error (a missed formulation opportunity), not a cosmetic one.
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
    fig, ax = plt.subplots(figsize=(max(10.0, len(models) * 1.3), 6.0))

    _shade_group_zones(ax, len(classical_sorted), len(models))

    ax.bar(x - width, f1, width, label="F1 score (primary)", color="#1B3B6F", zorder=3)
    ax.bar(x, recall1, width, label="Recall — Responder (recall_1)", color="#E67E22", zorder=3)
    ax.bar(x + width, acc, width, label="Accuracy (reference only)", color="#BDBDBD", zorder=3)

    if classical_sorted:
        best_classical_f1 = f1_of(classical_sorted[0])
        ax.axhline(best_classical_f1, color=COLOR_PATZMANN, linestyle="--", linewidth=1.3, zorder=4)
        # Anchored at the left edge (not right) so it never collides with
        # the "upper right" legend; a white backing box keeps it legible
        # even when it lands close to a bar-top value annotation.
        ax.text(
            -0.4, best_classical_f1 + 0.015, f"Classical baseline (F1={best_classical_f1:.3f})",
            ha="left", va="bottom", fontsize=TICK_FONTSIZE, style="italic", zorder=5,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.5),
        )

    # Annotate the top-3 quantum models (by F1) directly on their bars.
    top3_quantum = sorted(quantum_sorted, key=f1_of, reverse=True)[:3]
    for name in top3_quantum:
        idx = models.index(name)
        val = f1_of(name)
        ax.text(
            idx - width, val + 0.015, f"{val:.3f}", ha="center", va="bottom",
            fontsize=TICK_FONTSIZE, fontweight="bold", color="#1B3B6F",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right", fontsize=TICK_FONTSIZE)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontsize=LABEL_FONTSIZE)
    ax.set_title("LOOCV Classification Results — Classical vs Quantum (N=29)", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax.legend(fontsize=TICK_FONTSIZE, loc="upper right", framealpha=0.9)
    _style_axis(ax)
    fig.tight_layout()
    _save_fig(fig, output_dir, "classification_comparison")


def plot_roc_comparison(
    results: dict[str, dict],
    classical_models: list[str],
    quantum_models: list[str],
    output_dir: Path,
) -> None:
    """ROC curves for every classifier, classical and quantum, on one axis
    -- the single clearest "did quantum actually win" figure in the whole
    comparison suite: a curve that bows further toward the top-left corner,
    with a higher AUC in its own legend entry, is the better classifier at
    every operating threshold simultaneously, not just at the default 0.5
    cutoff `classification_comparison` reports.

    Built directly from each model's pooled LOOCV out-of-fold predictions
    (`targets`/`scores` in its JSON) -- the standard way to draw an ROC
    curve under leave-one-out cross-validation, since no single fold has
    enough held-out points for its own curve.

    Color identifies the model (fixed 8-slot categorical order, classical
    models first); line style identifies the family (solid = classical,
    dashed = quantum) as a second, color-independent encoding, so the two
    groups are still distinguishable in grayscale or under color
    blindness. The legend is sorted by AUC descending and doubles as the
    direct-label layer this many overlapping lines need.
    """
    classical_present = [m for m in classical_models if m in results]
    quantum_present = [m for m in quantum_models if m in results]
    models = classical_present + quantum_present
    if not models:
        print("[plot_roc_comparison] no models found in results, skipping")
        return

    # Compute every curve and its AUC once up front -- used for the plot,
    # the legend order, the bold "best model" line weight, and the saved
    # table-view twin, so nothing is recomputed.
    curves = {}
    for name in models:
        res = results[name]
        y_true = np.asarray(res["targets"], dtype=float)
        y_score = np.asarray(res["scores"], dtype=float)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        curves[name] = {
            "fpr": fpr, "tpr": tpr, "auc": roc_auc_score(y_true, y_score),
            "family": "classical" if name in classical_present else "quantum",
        }
    best_model = max(curves, key=lambda m: curves[m]["auc"])

    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    ax.plot([0, 1], [0, 1], color=COLOR_MUTED, linestyle="--", linewidth=1.1, zorder=1, label="Chance (AUC=0.500)")

    lines_by_model = {}
    for i, name in enumerate(models):
        c = curves[name]
        color = PALETTE_CATEGORICAL[i % len(PALETTE_CATEGORICAL)]
        (line,) = ax.plot(
            c["fpr"], c["tpr"], color=color, linestyle="-" if c["family"] == "classical" else "--",
            linewidth=2.6 if name == best_model else 1.6, zorder=3,
            label=f"{name} (AUC={c['auc']:.3f})", solid_capstyle="round", dash_capstyle="round",
        )
        lines_by_model[name] = line

    # Legend sorted by AUC descending (best model first), chance line last --
    # so the ranking that answers "which model actually wins" is readable
    # top-to-bottom without cross-referencing the plot.
    ranked_models = sorted(models, key=lambda m: -curves[m]["auc"])
    handles = [lines_by_model[m] for m in ranked_models] + [ax.lines[0]]
    labels = [f"{m} (AUC={curves[m]['auc']:.3f})" for m in ranked_models] + ["Chance (AUC=0.500)"]
    ax.legend(
        handles, labels, fontsize=TICK_FONTSIZE, loc="lower right", framealpha=0.92,
        title="Solid = classical · Dashed = quantum", title_fontsize=TICK_FONTSIZE - 1,
    )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False Positive Rate", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("True Positive Rate", fontsize=LABEL_FONTSIZE)
    ax.set_title("ROC — Classical vs Quantum Classifiers (LOOCV, N=29)", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax.set_aspect("equal")
    _style_axis(ax)
    fig.tight_layout()
    _save_fig(fig, output_dir, "roc_comparison")

    # Table-view twin: the exact AUC every curve above corresponds to.
    auc_table = pd.DataFrame(
        [{"model": m, "family": curves[m]["family"], "auc": curves[m]["auc"]} for m in ranked_models]
    )
    auc_table.to_csv(output_dir / "roc_auc_summary.csv", index=False)
    print(f"Saved {output_dir / 'roc_auc_summary.csv'}")


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


def plot_regression_comparison(
    results: dict[str, dict],
    classical_models: list[str],
    quantum_models: list[str],
    output_dir: Path,
) -> None:
    """Grouped bar chart: Q^2 (LOOCV, primary, left axis, green) and RMSE
    (secondary, right axis, red, COMDR15 units) for every regression
    model, sorted by Q^2 descending.

    What to look for: the Patzmann Q^2=0.77 dashed line is the single
    number every bar is implicitly judged against -- a bar crossing it
    matches or beats the published classical chemometric benchmark on this
    exact dataset. The Q^2=0 line matters just as much: a bar sitting at
    or below it means that model does no better (or worse) than always
    predicting the training-fold mean, no matter how small its RMSE looks
    in isolation (RMSE alone doesn't reveal whether a model is actually
    modeling the data or just regressing to its mean).
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

    # Q^2 and RMSE are different units on different scales -- a dual-axis
    # (twinx) bar chart would let their arbitrary relative scaling invent a
    # visual "RMSE goes up as Q^2 goes down" correlation that isn't
    # actually in the data. Two panels sharing the same model order and
    # x-axis instead: each bar height is only ever compared against other
    # bars on its own, honest scale.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(11.0, len(models) * 2.2), 5.6))

    _shade_group_zones(ax1, len(classical_sorted), len(models))
    ax1.bar(x, q2, width=0.6, color=PALETTE_CATEGORICAL[3], zorder=3)  # green: "higher is better"
    for xi, val in zip(x, q2):
        # A bar landing within 0.05 of the Pätzmann line (PLS does, by
        # construction) gets its value label pushed further up so it
        # clears both the dashed line and its own reference-label text box
        # instead of sitting on top of them.
        near_benchmark = val >= 0 and abs(val - PATZMANN_Q2_BENCHMARK) < 0.05
        offset = 0.11 if near_benchmark else (0.02 if val >= 0 else -0.05)
        ax1.text(
            xi, val + offset, f"{val:.3f}",
            ha="center", va="bottom" if val >= 0 else "top", fontsize=TICK_FONTSIZE, fontweight="bold",
        )
    ax1.axhline(PATZMANN_Q2_BENCHMARK, color=COLOR_PATZMANN, linestyle="--", linewidth=1.3, zorder=4)
    ax1.text(
        -0.45, PATZMANN_Q2_BENCHMARK + 0.05, f"Pätzmann benchmark (PLS, Q²={PATZMANN_Q2_BENCHMARK:.2f})",
        ha="left", va="bottom", fontsize=TICK_FONTSIZE - 1, style="italic", zorder=5,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.5),
    )
    ax1.axhline(0.0, color=COLOR_MUTED, linestyle=":", linewidth=1.1, zorder=4)
    ax1.text(
        -0.45, 0.02, "Baseline (predict mean)",
        ha="left", va="bottom", fontsize=TICK_FONTSIZE - 2, color=COLOR_MUTED, style="italic", zorder=5,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.5),
    )
    ax1.set_ylabel("Q² (LOOCV) — higher is better", fontsize=LABEL_FONTSIZE)
    ax1.set_ylim(min(-0.5, min(q2) - 0.15), 1.0)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=20, ha="right", fontsize=TICK_FONTSIZE)
    ax1.set_title("Q² vs Pätzmann benchmark", fontsize=TITLE_FONTSIZE - 1, fontweight="bold")
    _style_axis(ax1)

    _shade_group_zones(ax2, len(classical_sorted), len(models))
    ax2.bar(x, rmse, width=0.6, color=PALETTE_CATEGORICAL[5], zorder=3)  # red: "lower is better"
    for xi, val in zip(x, rmse):
        ax2.text(xi, val + max(rmse) * 0.015, f"{val:.2f}", ha="center", va="bottom", fontsize=TICK_FONTSIZE, fontweight="bold")
    ax2.set_ylabel("RMSE (COMDR15 units) — lower is better", fontsize=LABEL_FONTSIZE)
    ax2.set_ylim(0, max(rmse) * 1.15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=20, ha="right", fontsize=TICK_FONTSIZE)
    ax2.set_title("RMSE (physical COMDR15 units)", fontsize=TITLE_FONTSIZE - 1, fontweight="bold")
    _style_axis(ax2)

    fig.suptitle("LOOCV Regression Results — Classical vs Quantum (N=29)", fontsize=TITLE_FONTSIZE, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
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

    What to look for: points on the black y=x diagonal are perfect LOOCV
    predictions; distance from it is per-drug error. Point color follows
    the *classification* threshold (COMDR15=2.0) even in this regression
    view: green means the model landed on the correct side of the
    Responder/Non-Responder boundary regardless of its exact numeric
    error, red means it crossed to the wrong side -- a quick way to see
    whether a regression model's mistakes are at least directionally
    safe. Labeled points are drugs with |predicted - true| > 1.5,
    worth cross-referencing against physicochemical outliers.
    """
    candidates = [(classical_model, "Classical (PLS)"), (best_quantum_model, "Best Quantum")]
    models_to_plot = [(m, label) for m, label in candidates if m in results]
    if not models_to_plot:
        print("[plot_per_drug_regression_scatter] no models found in results, skipping")
        return

    fig, axes = plt.subplots(1, len(models_to_plot), figsize=(7.2 * len(models_to_plot), 6.5), squeeze=False)
    axes = axes[0]

    for ax, (model_name, label) in zip(axes, models_to_plot):
        res = results[model_name]
        y_true = np.asarray(res["targets"], dtype=float)
        y_pred = np.asarray(res["predictions"], dtype=float)
        m = res["metrics"]

        correct_side = (y_true > CLASSIFICATION_THRESHOLD) == (y_pred > CLASSIFICATION_THRESHOLD)
        colors = np.where(correct_side, "#0ca30c", "#d03b3b")  # status palette: good / critical

        ax.scatter(y_true, y_pred, c=colors, s=60, edgecolor="k", linewidth=0.5, zorder=3)

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
        ax.set_title(f"{label} ({model_name})\nQ²={m['q2']:.3f}, RMSE={m['rmse']:.3f}", fontsize=TITLE_FONTSIZE - 1)
        ax.legend(fontsize=TICK_FONTSIZE - 1, loc="upper left")
        _style_axis(ax, y_grid=False)

    fig.suptitle(
        "Per-Drug LOOCV Predictions — Best Quantum Regressor vs PLS Baseline",
        fontsize=TITLE_FONTSIZE, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save_fig(fig, output_dir, "per_drug_regression_scatter")


def plot_per_drug_classification_heatmap(
    results: dict[str, dict],
    drug_names: list[str],
    true_comdr15: np.ndarray,
    output_dir: Path,
) -> None:
    """(n_models x 29 drugs) heatmap: green if that model's LOOCV
    prediction for that drug was correct, red if wrong. Rows sorted by F1
    (best model at top), columns sorted by true COMDR15 ascending, with a
    vertical divider at the Responder/Non-Responder boundary.

    What to look for: a column that is mostly red straight down, across
    every row regardless of model family or paradigm, identifies a
    genuinely hard drug -- most likely one sitting close to the
    COMDR15=2.0 threshold or an outlier in feature space -- rather than a
    weakness specific to any one model. That is the direct visual answer
    to "which drugs are hardest to predict across all models" (Task 5).
    """
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

    fig, ax = plt.subplots(figsize=(max(12.0, len(order) * 0.5), max(4.0, len(models) * 0.6)))
    cmap = ListedColormap(["#d03b3b", "#0ca30c"])  # status palette: 0 -> wrong (critical), 1 -> correct (good)
    sns.heatmap(
        correctness, cmap=cmap, cbar=False, linewidths=0.6, linecolor="white",
        xticklabels=[f"{d}\n{c:.2f}" for d, c in zip(sorted_drugs, sorted_comdr)],
        yticklabels=models, ax=ax, vmin=0, vmax=1,
    )

    responder_boundary = int(np.searchsorted(sorted_comdr, CLASSIFICATION_THRESHOLD))
    ax.axvline(responder_boundary, color="black", linewidth=2.2)
    if responder_boundary > 0:
        ax.text(responder_boundary / 2, -0.6, "Non-Responder", ha="center", fontsize=TICK_FONTSIZE, fontweight="bold")
    if responder_boundary < len(order):
        ax.text(
            responder_boundary + (len(order) - responder_boundary) / 2, -0.6, "Responder",
            ha="center", fontsize=TICK_FONTSIZE, fontweight="bold",
        )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center", fontsize=TICK_FONTSIZE - 2)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=TICK_FONTSIZE)
    ax.set_title(
        "Per-Drug Prediction Correctness Across All Classification Models",
        fontsize=TITLE_FONTSIZE, fontweight="bold",
    )
    fig.tight_layout()
    _save_fig(fig, output_dir, "per_drug_classification_heatmap")


def plot_kta_comparison(results_dir: Path, output_dir: Path) -> None:
    """Horizontal bar chart: raw Kernel Target Alignment for the fixed
    (untrained) feature map encodings, next to our KTA-optimized trained
    kernel's alignment before vs. after COBYLA training.

    Reads two small CSVs `main.py` writes as a side effect of the
    quantum-classification and bonus stages:
      - kernel_kta_by_feature_map.csv: columns feature_map, kta -- the
        Task 3 diagnostic KTA for each fixed encoding (angle/entangled/zz).
      - kta_optimization.csv: columns kta_before, kta_after -- the trained
        kernel's KTA against a random initial weight vector vs. after
        COBYLA optimization (see TrainedQuantumKernelSVM in
        src/quantum_models.py).
    Skips with a printed message (not an error) if either file is missing,
    so a partial run (e.g. quantum-classification without bonus, or vice
    versa) still produces whatever half of this comparison it can.

    What to look for: this is the direct evidence for "training the kernel
    helps" -- the fixed encodings show KTA is a fairly weak, un-optimized
    property of an arbitrary circuit choice, while the trained kernel's
    bar visibly grows from its own (similarly weak) random starting point
    once COBYLA optimizes its weights against the training labels.
    """
    fm_path = Path(results_dir) / "kernel_kta_by_feature_map.csv"
    opt_path = Path(results_dir) / "kta_optimization.csv"
    rows = []
    if fm_path.exists():
        fm_df = pd.read_csv(fm_path)
        for _, r in fm_df.iterrows():
            rows.append({"label": f"Fixed: {r['feature_map']}", "kta": float(r["kta"]), "kind": "fixed"})
    else:
        print(f"[plot_kta_comparison] {fm_path} not found, skipping the fixed-encoding bars")

    if opt_path.exists():
        opt_df = pd.read_csv(opt_path)
        before, after = float(opt_df["kta_before"].iloc[0]), float(opt_df["kta_after"].iloc[0])
        rows.append({"label": "Trained kernel (before)", "kta": before, "kind": "trained_before"})
        rows.append({"label": "Trained kernel (after)", "kta": after, "kind": "trained_after"})
    else:
        print(f"[plot_kta_comparison] {opt_path} not found, skipping the trained-kernel bars")

    if not rows:
        print("[plot_kta_comparison] no KTA data found in results_dir, skipping")
        return

    table = pd.DataFrame(rows)
    color_by_kind = {
        "fixed": COLOR_MUTED,
        "trained_before": PALETTE_CATEGORICAL[7] + "80",  # orange, translucent: not yet optimized
        "trained_after": PALETTE_CATEGORICAL[7],           # orange, solid: the optimized result
    }
    colors = [color_by_kind[k] for k in table["kind"]]

    fig, ax = plt.subplots(figsize=(8.5, 1.1 + 0.7 * len(table)))
    y = np.arange(len(table))
    ax.barh(y, table["kta"], color=colors, height=0.6, zorder=3)
    for yi, val in zip(y, table["kta"]):
        ax.text(val + 0.012, yi, f"{val:.3f}", va="center", fontsize=TICK_FONTSIZE, fontweight="bold")

    # An arrow from before -> after, annotated with the exact improvement,
    # if both trained-kernel rows are present -- the one number this whole
    # figure exists to make impossible to miss.
    before_idx = table.index[table["kind"] == "trained_before"]
    after_idx = table.index[table["kind"] == "trained_after"]
    if len(before_idx) and len(after_idx):
        bi, ai = before_idx[0], after_idx[0]
        b_val, a_val = table["kta"].iloc[bi], table["kta"].iloc[ai]
        ax.annotate(
            "", xy=(a_val, y[ai]), xytext=(b_val, y[bi]),
            arrowprops=dict(arrowstyle="->", color=COLOR_PATZMANN, linewidth=1.4,
                             connectionstyle="arc3,rad=0.35"),
        )
        ax.text(
            max(b_val, a_val) + 0.09, (y[bi] + y[ai]) / 2,
            f"+{a_val - b_val:.3f} from training", fontsize=TICK_FONTSIZE, fontweight="bold",
            color=COLOR_PATZMANN, va="center",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(table["label"], fontsize=TICK_FONTSIZE)
    ax.invert_yaxis()  # first row (first fixed encoding) at the top
    ax.set_xlim(0, max(0.05, float(table["kta"].max())) * 1.35)
    ax.set_xlabel("Kernel Target Alignment (higher = kernel geometry matches labels better)", fontsize=LABEL_FONTSIZE)
    ax.set_title("Kernel Target Alignment — Fixed Encodings vs. Our Trained Kernel", fontsize=TITLE_FONTSIZE, fontweight="bold")
    _style_axis(ax, y_grid=False)
    ax.xaxis.grid(True, color=COLOR_GRID, linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save_fig(fig, output_dir, "kta_comparison")

    table.drop(columns="kind").to_csv(output_dir / "kta_comparison_table.csv", index=False)
    print(f"Saved {output_dir / 'kta_comparison_table.csv'}")


def main(argv: list[str] | None = None) -> None:
    """argv defaults to None, which makes argparse read sys.argv[1:] as
    usual for `python scripts/plot_model_comparison.py ...`. Callers that
    import and invoke this directly (main.py, after a full pipeline run)
    pass argv=[] explicitly instead, so this never tries to parse main.py's
    own --stage/--backend/etc. flags as its own arguments.
    """
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
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir)

    clf_results = load_results(args.results_dir, CLASSIFICATION_CLASSICAL_MODELS + CLASSIFICATION_QUANTUM_MODELS)
    reg_results = load_results(args.results_dir, REGRESSION_CLASSICAL_MODELS + REGRESSION_QUANTUM_MODELS)

    plot_classification_comparison(clf_results, CLASSIFICATION_CLASSICAL_MODELS, CLASSIFICATION_QUANTUM_MODELS, output_dir)
    plot_roc_comparison(clf_results, CLASSIFICATION_CLASSICAL_MODELS, CLASSIFICATION_QUANTUM_MODELS, output_dir)
    plot_confusion_matrices(
        clf_results, CLASSIFICATION_CLASSICAL_MODELS, output_dir, "Classical ML", COLOR_CLASSICAL_ZONE, "confusion_matrices_classical"
    )
    plot_confusion_matrices(
        clf_results, CLASSIFICATION_QUANTUM_MODELS, output_dir, "Quantum ML", COLOR_QUANTUM_ZONE, "confusion_matrices_quantum"
    )
    plot_regression_comparison(reg_results, REGRESSION_CLASSICAL_MODELS, REGRESSION_QUANTUM_MODELS, output_dir)

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
        plot_per_drug_classification_heatmap(clf_results, drug_names, true_comdr15, output_dir)

    plot_kta_comparison(Path(args.results_dir), output_dir)


if __name__ == "__main__":
    main()
