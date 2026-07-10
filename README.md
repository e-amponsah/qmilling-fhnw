# OQI Hackathon 2026 — Predicting Drug Dissolution Enhancement with (Quantum) ML

A production-grade classical + quantum machine learning pipeline for the OQI Hackathon 2026
challenge: predicting whether co-milling a crystalline drug with PVP K25 will meaningfully
enhance its dissolution rate, from molecular structure (SMILES) and experimental particle size
alone. See [`OQI_Hackathon_2026.pdf`](OQI_Hackathon_2026.pdf) for the full challenge brief and
[Pätzmann et al. 2024](https://doi.org/10.1016/j.ejps.2024.106780) for the source dataset and
classical chemometric benchmark (Q² = 0.77).

Every quantum computation in this repo — kernel evaluation, variational-model training,
inference — runs as a **real Qiskit `SamplerV2` job**, shot-sampled from a transpiled circuit.
There is no `Statevector`/exact-linear-algebra shortcut anywhere in the model code. The same
code path runs on a local `AerSimulator` or on real IBM Quantum hardware via
`qiskit-ibm-runtime` — it's a one-flag toggle (`--backend`).

---

## Contents

- [Quickstart](#quickstart)
- [Repository layout](#repository-layout)
- [The problem, in brief](#the-problem-in-brief)
- [Pipeline stages](#pipeline-stages)
- [Quantum execution: local simulator vs. real IBM Quantum hardware](#quantum-execution-local-simulator-vs-real-ibm-quantum-hardware)
- [Models](#models)
- [Evaluation methodology & leakage guarantees](#evaluation-methodology--leakage-guarantees)
- [Bonus extensions](#bonus-extensions)
- [CLI reference](#cli-reference)
- [Notebooks](#notebooks)
- [Known limitations & deliberate scope choices](#known-limitations--deliberate-scope-choices)
- [Troubleshooting](#troubleshooting)

---

## Quickstart

```bash
# 1. Create and activate a virtual environment (Python 3.11+)
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) enable real IBM Quantum hardware execution
cp .env.example .env          # then fill in IBM_QUANTUM_CRN / IBM_QUANTUM_API_KEY
python scripts/setup_ibm_account.py

# 4. Run the full pipeline (local simulator by default -- no IBM account needed)
python main.py --stage all
```

Outputs land in `data/processed/`, `data/results/`, and `plots/` (see below). A full run takes
roughly 20–30 minutes on a laptop CPU, almost entirely spent in the quantum LOOCV suite (Stage
3) — real circuit execution, even against a local ideal simulator, is inherently slower than an
exact-statevector shortcut, by design (see the top of this file).

---

## Repository layout

```
├── data/
│   ├── raw/                  # PubChem SMILES lookup + the provided experimental dataset
│   ├── processed/             # RDKit descriptor table + merged modeling table
│   └── results/               # LOOCV result tables (classical, quantum, unified, bonus)
├── plots/
│   ├── circuits/               # Circuit diagrams for every feature map / ansatz / QCNN
│   ├── kernels/                 # 29x29 quantum kernel heatmaps, per feature map
│   ├── pca/                      # Scree, PC1-PC2 scatter, biplot, PC-vs-target correlation
│   ├── scatter/                  # Per-drug predicted score vs. true COMDR_15min
│   ├── feature_correlation_heatmap.png
│   └── unified_comparison.png     # Classical vs. quantum LOOCV accuracy bar chart
├── notebooks/                # Thin, interactive wrappers around src/ -- no logic lives here
├── scripts/
│   └── setup_ibm_account.py  # One-time IBM Cloud account setup from .env
├── src/
│   ├── config.py              # Paths, column contracts, constants -- single source of truth
│   ├── data_fetching.py        # PubChem REST SMILES lookup, rate-limited & fault-tolerant
│   ├── features.py              # RDKit descriptors + fold-safe, PCA-guided k-best selection
│   ├── pca_analysis.py           # PCA diagnostic + redundancy-cluster detection
│   ├── quantum_circuits.py       # Circuit *layouts*: angle/entangled/ZZ feature maps, VQC ansatz
│   ├── quantum_backend.py         # Real execution layer: local Aer <-> IBM Runtime toggle
│   ├── quantum_models.py           # Executable classes: QK-SVM, VQC, Data Re-upload, QCNN
│   ├── classical_models.py          # SVC / RandomForest / GradientBoosting / PLS baselines
│   ├── evaluation.py                 # LOOCV engine, KTA metric, leakage-guard self-test
│   └── bonus_extensions.py            # KTA optimization, noise/hardware study, blind predictor
├── main.py                    # CLI entry point orchestrating every stage
├── requirements.txt
├── .env.example                # Template for IBM Cloud credentials (copy to .env)
└── OQI_Hackathon_2026.pdf      # The original challenge brief
```

---

## The problem, in brief

29 structurally diverse, crystalline drugs were co-milled with PVP K25 and characterized by
in-vitro dissolution testing (Pätzmann et al. 2024). The key response variable,

```
COMDR_15min = COM_15min / PM_15min
```

(co-milled vs. physical-mixture dissolved concentration at 15 minutes), is thresholded at 2.0 to
define a binary label: `1` = Responder (co-milling at least doubles dissolution), `0` =
Non-Responder. **13 of 29 drugs are Responders.**

**Critical leakage rule** (enforced in `src/config.py::LEAKAGE_COLS` and asserted in
`src/features.py::get_candidate_matrix`): `COM_15min` and `PM_15min` mathematically define the
target and must never appear in the feature matrix `X`. The candidate feature pool is strictly
the 14 RDKit descriptors + the two experimental columns `D50` (particle size) and
`apparent_solubility` (Sapp, apparent solubility in FaSSIF) — 16 columns total.

Pätzmann et al.'s own model used 4 variables: `D50`, `logD6.5`, `κ3`, `Sapp`. All four are
reproducible from the provided data: `D50` and `apparent_solubility` (Sapp) directly, `Kappa3`
directly as an RDKit topological descriptor, and `MolLogP` as the standard approximation for
`logD6.5` (RDKit doesn't compute a pH-dependent logD).

---

## Pipeline stages

Run any stage independently with `python main.py --stage <name>`; `all` runs every stage in
order and produces the final unified comparison.

| Stage | What it does | Key output |
|---|---|---|
| `fetch` | PubChem REST lookup of all 29 SMILES (rate-limited ≥0.34s/request, RDKit-validated) | `data/raw/smiles_29_drugs.csv` |
| `features` | Compute 14 RDKit descriptors, merge with experimental data, PCA diagnostic, fold-safe PCA-guided SelectKBest sweep (k=4..6) | `data/processed/modeling_table.csv`, `data/results/selected_features.csv`, `data/results/pca_loadings.csv`/`pca_scores.csv`, `plots/feature_correlation_heatmap.png`, `plots/pca/*` |
| `classical` | ≥2 leakage-free classical baselines (SVC, RandomForest, GradientBoosting) + a PLS regression baseline for direct Pätzmann Q² comparison, all under 29-fold LOOCV | `data/results/classical_loocv.csv` |
| `quantum` | Circuit diagrams, 3 kernel heatmaps + KTA scores, and the full 5-model quantum suite (QK-SVM×2, VQC, Data Re-uploading, QCNN) under 29-fold LOOCV | `data/results/quantum_loocv.csv`, `plots/circuits/*`, `plots/kernels/*`, `plots/scatter/score_vs_comdr.png` |
| `bonus` | KTA-optimized ("trained") quantum kernel, ideal-vs-noisy/hardware degradation study, blind SMILES prediction demo | `data/results/execution_degradation.csv` |
| `all` | Everything above + unified classical-vs-quantum comparison | `data/results/unified_comparison.csv`, `plots/unified_comparison.png` |

The QCNN is architecturally fixed at 6 qubits (per the challenge spec), so it always runs on its
own dedicated 6-feature subset rather than whichever k the automated selector found optimal for
the other models — see `main.py::stage_quantum`.

### PCA diagnostic & redundancy-aware feature selection

`src/pca_analysis.py` runs PCA on the full 16-column candidate pool as a **diagnostic**, not a
feature-reduction step — every model still trains on physically-interpretable RDKit descriptors,
never abstract PCA components. On this dataset, PC1 alone explains ~60% of variance (4 PCs are
needed to cover 80%, 7 for 95%): `MolWt`, `Chi0v`, `Chi1v`, `Kappa1-3`, `BertzCT`, `TPSA`,
`NumHAcceptors/Donors`, `NumRotatableBonds`, `RingCount`, and `apparent_solubility` all load
heavily onto it — they're mostly measuring the same latent "molecular size/shape" axis (see
`plots/pca/pca_biplot.png`, where those arrows all point the same direction). `MolLogP` and
`FractionCSP3` share PC2. `D50` turns out to be entirely alone on PC3 — the only feature that's
an independent experimental particle-size measurement rather than derived from molecular
structure or solubility — which explains why it correlates strongest with the target yet is easy
for a naive selector to overlook.

`select_k_best_features` uses this: instead of blindly taking the top-k individually-scoring
(`f_classif`) features, it takes at most **one representative per PCA-identified cluster** first,
falling back to raw score only once every cluster has contributed a feature
(`pca_analysis.redundancy_aware_topk`). On this dataset that took the nested-LOOCV selection
score from ~0.69 to ~0.86 and pulled `D50` into the selected set — plain top-k `SelectKBest` had
been picking 3-4 mutually-redundant size-cluster descriptors instead. Both the nested-LOOCV
scoring loop *and* the final full-dataset selection use PCA fit strictly within their own
training data, so this redundancy guard never leaks the held-out sample either.

Deliverables: `plots/pca/pca_scree.png` (variance per PC + cumulative), `pca_scatter.png`
(drugs in PC1-PC2 space, colored by Responder/Non-Responder), `pca_biplot.png` (loading arrows +
drug scores together), `pca_target_corr.png` (each PC's correlation with `COMDR_15min`), plus
`data/results/pca_loadings.csv` / `pca_scores.csv` for the full numeric tables.

### Manual feature override

`selected_features.csv` is chosen by an *automated* statistical filter (fold-safe
`SelectKBest(f_classif)` against the binary label) — it is **not** the same thing as "the
features most correlated with `COMDR_15min`" shown in `plots/feature_correlation_heatmap.png`
(that heatmap correlates against the *continuous* target and is purely for visual inspection).
The two can disagree — e.g. `D50` has by far the strongest correlation with `COMDR_15min` (0.81)
but isn't picked by the classification-oriented automated filter.

If you've inspected the heatmap yourself and want the pipeline to use your own hand-picked
columns for every model instead, set `MANUAL_FEATURES` in `.env` (comma-separated, must be a
subset of `src/config.py::CANDIDATE_FEATURE_COLS` — the 14 RDKit descriptors + `D50`):

```bash
# .env
MANUAL_FEATURES=D50,MolLogP,Kappa3,TPSA
```

This bypasses `select_k_best_features` entirely (`src/features.py::get_modeling_features` is the
single entry point every stage calls, and it checks `MANUAL_FEATURES` first). If your manual list
isn't exactly 6 features, the fixed 6-qubit QCNN automatically falls back to its own automated
6-best subset — logged explicitly — since it structurally cannot accept any other qubit count.
Leave `MANUAL_FEATURES` empty/unset to keep the automated selection.

---

## Quantum execution: local simulator vs. real IBM Quantum hardware

Every quantum model in `src/quantum_models.py` takes a live `QuantumExecutor`
(`src/quantum_backend.py`), which resolves **once** per suite run to either:

| `--backend` flag | `ExecutionConfig.mode` | What runs |
|---|---|---|
| `aer` (default) | `aer_simulator` | Local `AerSimulator`, ideal (noiseless) |
| `aer-noisy` | `aer_noisy` | Local `AerSimulator` + a synthetic device-like noise model (depolarizing gate errors + readout error) |
| `ibm-runtime` | `ibm_runtime` | A real (or cloud-hosted) IBM Quantum backend via `QiskitRuntimeService` |

All three paths use the exact same `SamplerV2`-based circuit-execution code
(`QuantumExecutor.run_counts_batch`) — the only thing that changes is which `Backend` object the
Sampler is pointed at. Every circuit needed for one LOOCV fold (or one optimizer iteration) is
batched into a **single** Sampler job, which is what makes real-hardware/cloud-queue execution
remotely practical instead of submitting one job per circuit.

### Enabling IBM Quantum hardware

1. Get your **Cloud Resource Name (CRN)** and an **API key** from the IBM Cloud console → your
   Qiskit Runtime service instance → "Manage" tab (API key: IAM → API keys → Create).
2. `cp .env.example .env` and fill in `IBM_QUANTUM_CRN` and `IBM_QUANTUM_API_KEY`.
3. `python scripts/setup_ibm_account.py` — saves the account (`channel="ibm_cloud"`) to the
   standard `qiskit-ibm-runtime` credential store and verifies it by listing available backends.
4. `python main.py --stage quantum --backend ibm-runtime --max-samples 8` — **strongly
   recommended before a full run**: a full 29-fold LOOCV across 5 quantum models submits
   hundreds of jobs to a real, queued device. `--max-samples` caps the LOOCV sample count for a
   cheap smoke test first.

`QC_BACKEND_MODE`, `QC_SHOTS`, and `QC_IBM_BACKEND` in `.env` set the same things as `--backend`
/ `--shots` / `--ibm-backend` without needing to pass CLI flags every time
(`src/quantum_backend.py::default_execution_config`).

**Cost/time reality check**: local `aer` execution of the full 5-model, 29-fold LOOCV suite
takes ~20–25 minutes on a laptop. The same suite against `ibm-runtime` would take dramatically
longer (real device queue times per job) and consume real device time — use `--max-samples` and/or
`model_names=[...]` to scope real-hardware runs deliberately rather than defaulting to the full
sweep.

---

## Models

### Classical baselines (`src/classical_models.py`)
- **SVC (RBF kernel)**, calibrated via `CalibratedClassifierCV` for `predict_proba`.
- **RandomForestClassifier**, **GradientBoostingClassifier**.
- **PLS regression** on `COMDR_15min` (continuous target) for direct comparison against the
  Pätzmann Q²=0.77 benchmark.

### Quantum feature maps (`src/quantum_circuits.py`)
- `angle_feature_map` — one `Ry(x_i)` per qubit + a single CNOT chain (the challenge's baseline
  encoding, Eq. 3 of the brief).
- `entangled_feature_map` — the angle encoding repeated with a closed CNOT *ring* after each
  repetition, for deeper entanglement.
- `zz_feature_map` — Qiskit's `ZZFeatureMap` (angle encoding + pairwise ZZ interaction terms).

### Quantum models (`src/quantum_models.py`)
1. **QK-SVM** (`QuantumKernelSVM`) — fidelity kernel `|⟨ψ(xᵢ)|ψ(xⱼ)⟩|²` measured via an actual
   compute-uncompute circuit (not `Statevector.inner`), fed to a classical `SVC(kernel=
   "precomputed")`. Registered for both the `angle` and `zz` feature maps.
2. **VQC** (`VariationalQuantumClassifier`) — angle encoding + a trainable Ry/Rz + CNOT-chain
   ansatz, BCE-trained.
3. **Data Re-uploading Classifier** — repeats [trainable rotation → data re-encoding →
   entanglement] `n_layers` times, giving universal approximation power without extra qubits.
4. **QCNN** (`QCNNClassifier`) — 6-qubit angle encoding → conv+pool (6→3) → conv+pool (3→2) → a
   full 15-parameter SU(4) dense layer on the 2 surviving qubits → single output qubit (~54
   trainable parameters total, in line with the brief's "~51").

All four variational architectures share one training core (`_VariationalCore`) with three
selectable, job-batched optimizers:
- `cobyla` (default) — gradient-free, one Sampler job per scalar-loss evaluation.
- `spsa` — one Sampler job of `2N` circuits per iteration (the ± perturbation pair batched
  together) — the standard choice for real-hardware variational training.
- `parameter_shift` — the exact analytic gradient, batched into one job of `2·n_params·N`
  circuits per iteration. Most accurate, most expensive; intended for small illustrative
  comparisons (see the brief's "compare parameter-shift to SPSA"), not the full LOOCV sweep.

---

## Evaluation methodology & leakage guarantees

- **Full 29-fold Leave-One-Out Cross-Validation** for every model (`src/evaluation.py`).
- **Per-fold scaling only**: `MinMaxScaler` is `.fit()` on the 28 training rows and only
  `.transform()`-ed on the held-out row — every single time, for every model, no exceptions.
  Quantum models scale to `[0, π]` (raw rotation angles); classical models to `[0, 1]`.
- **Feature selection is fold-safe by construction**: `select_k_best_features` runs its own
  nested LOOCV internally, fitting `VarianceThreshold` + `SelectKBest` only on each fold's
  training data before scoring the held-out sample.
- `src/evaluation.py` ships a runnable self-test (`python -m src.evaluation`) that documents and
  exercises the no-leak guarantee.
- Reported metrics: accuracy, F1, per-class recall (classification); Q² (LOOCV) and R² (training
  refit) for the PLS regression baseline.

---

## Bonus extensions (`src/bonus_extensions.py`)

- **KTA optimization** (`KTAOptimizedQuantumKernel`): a "trained" quantum kernel — the feature
  map's *weight* parameters (not the data-encoding parameters) are optimized via COBYLA to
  maximize Kernel Target Alignment against the training labels before the SVM head is fit.
- **Execution-degradation study** (`execution_degradation_study`): compares QK-SVM accuracy
  between an ideal baseline and a comparison execution target under leak-free K-fold CV. Defaults
  to ideal-vs-synthetic-noise; pass `comparison_config=ExecutionConfig(mode="ibm_runtime")` to run
  the identical comparison against genuine hardware with no other code changes.
- **Blind SMILES prediction interface** (`BlindPredictor`): `.predict(smiles, d50=...)` →
  `{"prediction": "Responder"/"Non-Responder", "confidence": float}`. Computes RDKit descriptors
  on the fly, applies an already-fitted (training-set-only) scaler, and never re-fits anything.
  `D50` is an experimental measurement, not derivable from SMILES — passing it is required
  whenever the deployed model's feature set includes it; omitting it raises rather than guessing.

---

## CLI reference

```
python main.py [--stage {fetch,features,classical,quantum,bonus,all}]
                [--backend {aer,aer-noisy,ibm-runtime}]
                [--shots N]
                [--ibm-backend NAME]
                [--max-samples N]
```

| Flag | Default | Meaning |
|---|---|---|
| `--stage` | `all` | Which pipeline stage(s) to run |
| `--backend` | `aer` | Where quantum circuits execute (see table above) |
| `--shots` | `4096` | Shots per circuit execution |
| `--ibm-backend` | *auto (least-busy)* | Pin a specific IBM backend name instead of auto-selection |
| `--max-samples` | *None (full 29)* | Cap the LOOCV sample count for the quantum/bonus stages — use for cheap smoke tests, especially before an `ibm-runtime` run |

---

## Notebooks

`notebooks/01_explore_data_and_features.ipynb`, `02_classical_and_quantum_models.ipynb`, and
`03_bonus_and_blind_prediction.ipynb` are thin interactive wrappers that call into `src/` — no
logic is duplicated there. Each notebook exposes an `EXECUTION_CONFIG`/`ExecutionConfig` cell so
the backend can be switched the same way as the CLI's `--backend` flag.

---

## Known limitations & deliberate scope choices

- **Reduced optimizer iteration budgets for real execution**: VQC/Data-Reuploading default to
  `maxiter=60`, QCNN to `maxiter=80` (COBYLA) — tuned down from what would be affordable under an
  exact-statevector simulation, because every iteration is now a real, shot-sampled Sampler job.
  Increase `maxiter` in `src/quantum_models.py::QUANTUM_MODEL_BUILDERS` for higher-quality (but
  slower) training.
- **QCNN's fixed 6-qubit architecture** means it necessarily uses a different (not
  LOOCV-optimal) feature subset than the other 4-6-qubit models; this is called out explicitly in
  logs rather than silently forcing the QCNN onto a 4-feature subset it can't accept.
- **`ibm-runtime` mode is fully wired but not something this pipeline defaults to for the full
  LOOCV sweep** — real device queue time and cost make that impractical to run unattended; use
  `--max-samples` to scope real-hardware runs.

---
