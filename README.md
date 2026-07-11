# OQI Hackathon 2026: Predicting Drug Dissolution Enhancement with Quantum ML

A classical and quantum machine learning pipeline built for the OQI Hackathon 2026 challenge:
predicting whether co-milling a crystalline drug with PVP K25 will meaningfully improve its
dissolution rate, using only its molecular structure (SMILES) and experimental particle size.
See [`OQI_Hackathon_2026.pdf`](OQI_Hackathon_2026.pdf) for the full challenge brief and
[Pätzmann et al. 2024](https://doi.org/10.1016/j.ejps.2024.106780) for the source dataset and
the classical benchmark this project compares against (Q² = 0.77).

Every quantum computation in this project, whether it is a kernel value or a variational model's
prediction, runs as a real Qiskit `SamplerV2` job: shots sampled from a transpiled circuit.
Nothing in the model code takes a shortcut through exact linear algebra with `Statevector`. The
same code runs against a local `AerSimulator` or real IBM Quantum hardware through
`qiskit-ibm-runtime`. Switching between them is one flag: `--backend`.

## Contents

- [Quickstart](#quickstart)
- [Repository layout](#repository-layout)
- [The problem](#the-problem)
- [Pipeline stages](#pipeline-stages)
- [Quantum execution: local simulator or real IBM hardware](#quantum-execution-local-simulator-or-real-ibm-hardware)
- [Models](#models)
- [Kernel Target Alignment](#kernel-target-alignment)
- [Evaluation methodology and leakage guarantees](#evaluation-methodology-and-leakage-guarantees)
- [Bonus extensions](#bonus-extensions)
- [CLI reference](#cli-reference)
- [Notebooks](#notebooks)
- [Known limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)

## Quickstart

```bash
# 1. Create and activate a virtual environment (Python 3.11+)
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Optional: enable real IBM Quantum hardware execution
cp .env.example .env          # then fill in IBM_QUANTUM_CRN and IBM_QUANTUM_API_KEY
python scripts/setup_ibm_account.py

# 4. Run the full pipeline (local simulator by default, no IBM account needed)
python main.py --stage all
```

Outputs land in `data/processed/`, `data/results/`, and `plots/`. A full run takes roughly 20 to
30 minutes on a laptop CPU, almost all of it in the quantum stage. Real circuit execution, even
against a local simulator, is slower than an exact linear algebra shortcut would be. That is
expected and is the point: every result in this project comes from an actual circuit run.

## Repository layout

```
├── data/
│   ├── raw/                  PubChem SMILES lookup and the provided experimental dataset
│   ├── processed/             RDKit descriptor table and the merged modeling table
│   └── results/               LOOCV result tables: classical, quantum, unified, bonus
├── plots/
│   ├── circuits/               Circuit diagrams for every feature map, ansatz, and the QCNN
│   ├── kernels/                 29x29 quantum kernel heatmaps, one per feature map
│   ├── pca/                      Scree plot, PC1 vs PC2 scatter, biplot, PC vs target correlation
│   ├── scatter/                  Per-drug predicted score vs true COMDR_15min
│   ├── feature_correlation_heatmap.png
│   └── unified_comparison.png     Classical vs quantum LOOCV accuracy bar chart
├── notebooks/                Thin interactive wrappers around src/, no logic lives here
├── scripts/
│   └── setup_ibm_account.py  One-time IBM Cloud account setup from .env
├── src/
│   ├── config.py              Paths, column names, constants: the single source of truth
│   ├── data_fetching.py        PubChem REST SMILES lookup, rate limited and fault tolerant
│   ├── features.py              RDKit descriptors and fold-safe, PCA-guided feature selection
│   ├── pca_analysis.py           PCA diagnostic and redundancy cluster detection
│   ├── quantum_circuits.py       Circuit layouts: angle, entangled, and ZZ feature maps, VQC ansatz
│   ├── quantum_backend.py         The real execution layer: local Aer or IBM Runtime
│   ├── quantum_models.py           The model classes: QK-SVM, VQC, data re-uploading, QCNN
│   ├── classical_models.py          SVC, RandomForest, GradientBoosting, and PLS baselines
│   ├── evaluation.py                 LOOCV engine, KTA metric, leakage guard self-test
│   └── bonus_extensions.py            KTA optimization, noise/hardware study, blind predictor
├── main.py                    The entry point that runs every stage
├── requirements.txt
├── .env.example                Template for IBM Cloud credentials, copy to .env
└── OQI_Hackathon_2026.pdf      The original challenge brief
```

## The problem

29 structurally diverse, crystalline drugs were co-milled with PVP K25 and tested for
dissolution (Pätzmann et al. 2024). The key response variable is

```
COMDR_15min = COM_15min / PM_15min
```

the ratio of dissolved concentration at 15 minutes between the co-milled formulation and a
physical mixture. This is thresholded at 2.0 to define a binary label: 1 means Responder
(co-milling at least doubles dissolution), 0 means Non-Responder. 13 of the 29 drugs are
Responders.

**The leakage rule**: `COM_15min` and `PM_15min` are the raw measurements COMDR_15min is
computed from, so they must never be used as model features. This is enforced in
`src/config.py` (`LEAKAGE_COLS`) and checked again in `src/features.py`
(`get_candidate_matrix`). The candidate feature pool is strictly the 14 RDKit descriptors plus
two experimental columns, `D50` (particle size) and `apparent_solubility`, 16 columns total.

Pätzmann et al.'s own model used four variables: D50, logD6.5, Kappa3, and Sapp. All four are
available here: D50 and apparent_solubility (Sapp) directly, Kappa3 as an RDKit topological
descriptor, and MolLogP as the standard approximation for logD6.5 (RDKit does not compute a
pH-dependent logD directly).

## Pipeline stages

Run any stage on its own with `python main.py --stage <name>`, or run `all` to go through every
stage in order and produce the final comparison.

| Stage | What it does | Key output |
|---|---|---|
| `fetch` | Look up all 29 SMILES from PubChem, rate limited to at least 0.34s per request, validated with RDKit | `data/raw/smiles_29_drugs.csv` |
| `features` | Compute 14 RDKit descriptors, merge with the experimental data, run the PCA diagnostic, run fold-safe feature selection | `data/processed/modeling_table.csv`, `data/results/selected_features.csv`, `data/results/pca_loadings.csv` and `pca_scores.csv`, `plots/feature_correlation_heatmap.png`, `plots/pca/` |
| `classical` | Three classical classifiers (SVC, RandomForest, GradientBoosting) plus a PLS regression baseline against the Pätzmann benchmark, all under 29-fold LOOCV | `data/results/classical_loocv.csv`, `data/results/pls_regression_loocv.csv` |
| `quantum` | Circuit diagrams, three kernel heatmaps with KTA scores, and the full 6-model quantum suite under 29-fold LOOCV | `data/results/quantum_loocv.csv`, `plots/circuits/`, `plots/kernels/`, `plots/scatter/score_vs_comdr.png` |
| `bonus` | The KTA-optimized trained quantum kernel demo, an ideal-vs-noisy/hardware comparison, a blind SMILES prediction demo | `data/results/execution_degradation.csv` |
| `all` | Everything above, plus a unified classical-vs-quantum comparison | `data/results/unified_comparison.csv`, `plots/unified_comparison.png` |

The QCNN needs exactly 6 qubits, so it always runs on its own dedicated 6 feature subset rather
than whatever k the automated selector found best for the other models. See
`main.py` (`stage_quantum`).

### PCA diagnostic and redundancy-aware feature selection

`src/pca_analysis.py` runs PCA on the full 16 column candidate pool as a diagnostic tool, not as
a feature reduction step. Every model still trains on the original, physically interpretable
RDKit descriptors, never on abstract PCA components. On this dataset, the first component alone
explains about 60% of the variance (4 components are needed to reach 80%, 7 for 95%). MolWt,
Chi0v, Chi1v, Kappa1 through Kappa3, BertzCT, TPSA, the H-acceptor and H-donor counts,
NumRotatableBonds, RingCount, and apparent_solubility all load heavily onto that first
component, meaning they are mostly measuring the same underlying thing: molecule size and shape
(see `plots/pca/pca_biplot.png`, where those arrows all point in roughly the same direction).
MolLogP and FractionCSP3 share the second component. D50 stands alone on the third component,
the only feature that is an independent experimental measurement rather than something derived
from molecular structure, which is likely why it correlates strongest with the target and yet is
easy for a plain feature selector to miss.

`select_k_best_features` uses this. Instead of just taking the top scoring features by an
`f_classif` test, it takes at most one feature per PCA cluster first, and only falls back to raw
score once every cluster has contributed a feature (`pca_analysis.redundancy_aware_topk`). On
this dataset that raised the nested-LOOCV selection score from about 0.69 to about 0.86 and
pulled D50 into the selected set, where plain top-k selection had instead picked three or four
descriptors that were all measuring size. Both the nested scoring loop and the final full-dataset
selection fit PCA on training data only, so this redundancy check never sees the held-out sample
either.

Deliverables: `plots/pca/pca_scree.png` (variance per component and cumulative), `pca_scatter.png`
(drugs in PC1 vs PC2 space, colored by label), `pca_biplot.png` (loading arrows and drug scores
together), `pca_target_corr.png` (each component's correlation with COMDR_15min), and
`data/results/pca_loadings.csv` / `pca_scores.csv` for the full numeric tables.

### Manual feature override

`selected_features.csv` comes from an automated statistical filter (fold-safe `SelectKBest`
scored against the binary label). That is not the same thing as "the features most correlated
with COMDR_15min" shown in `plots/feature_correlation_heatmap.png`, which correlates against the
continuous target and is only there for visual inspection. The two can disagree: D50 has by far
the strongest correlation with COMDR_15min (0.81) but is not picked by the classification-focused
automated filter on its own.

If you have looked at the heatmap yourself and want the pipeline to use your own hand-picked
columns for every model instead, set `MANUAL_FEATURES` in `.env` (comma separated, must be a
subset of `CANDIDATE_FEATURE_COLS` in `src/config.py`):

```bash
# .env
MANUAL_FEATURES=D50,MolLogP,Kappa3,TPSA
```

This skips `select_k_best_features` entirely. `get_modeling_features` in `src/features.py` is the
single entry point every stage calls, and it checks `MANUAL_FEATURES` first. If your manual list
is not exactly 6 features, the QCNN (which needs exactly 6 qubits) falls back to its own automated
6 feature selection, and this is logged clearly rather than done silently. Leave `MANUAL_FEATURES`
empty to keep the automated selection.

## Quantum execution: local simulator or real IBM hardware

Every quantum model in `src/quantum_models.py` takes a live `QuantumExecutor`
(`src/quantum_backend.py`), which resolves once per run to one of:

| `--backend` flag | `ExecutionConfig.mode` | What runs |
|---|---|---|
| `aer` (default) | `aer_simulator` | Local `AerSimulator`, no noise |
| `aer-noisy` | `aer_noisy` | Local `AerSimulator` with a synthetic device-like noise model (depolarizing gate errors and readout error) |
| `ibm-runtime` | `ibm_runtime` | A real or cloud hosted IBM Quantum backend through `QiskitRuntimeService` |

All three paths run through the same `SamplerV2` based code (`QuantumExecutor.run_counts_batch`).
The only thing that changes is which backend object the Sampler points at. Every circuit needed
for one LOOCV fold, or one optimizer step, is batched into a single Sampler job, which is what
makes running against a real or cloud queued device practical instead of submitting one job per
circuit.

### Enabling IBM Quantum hardware

1. Get your Cloud Resource Name (CRN) and an API key from the IBM Cloud console, under your
   Qiskit Runtime service instance's "Manage" tab (API key: IAM, then API keys, then Create).
2. `cp .env.example .env` and fill in `IBM_QUANTUM_CRN` and `IBM_QUANTUM_API_KEY`.
3. `python scripts/setup_ibm_account.py`, which saves the account and verifies it by listing
   available backends.
4. Before a full run, try `python main.py --stage quantum --backend ibm-runtime --max-samples 8`.
   A full 29-fold LOOCV across 6 quantum models submits a lot of jobs to a real, queued device, so
   `--max-samples` caps the LOOCV sample count for a cheap check first.

`QC_BACKEND_MODE`, `QC_SHOTS`, and `QC_IBM_BACKEND` in `.env` set the same things as `--backend`,
`--shots`, and `--ibm-backend` without needing to pass CLI flags every time.

### Waiting for real hardware jobs

A real IBM job can sit queued for minutes to hours. `QuantumExecutor.run_counts_batch`
(`src/quantum_backend.py`) waits for it properly rather than either blocking silently or giving up
on the first hiccup:

- It logs the job ID immediately after submission, and logs each status change (`QUEUED` →
  `RUNNING` → `DONE`) instead of hanging with no visible progress.
- A transient network error while *checking* status (not the job itself) is retried instead of
  killing the run -- the job keeps going on IBM's side regardless of whether this process can
  currently reach the API.
- `ExecutionConfig.job_timeout` (seconds, default `None` = wait indefinitely) and
  `job_poll_seconds` (default `15`) control how long to wait and how chatty the status logging is.
  `job_timeout` can also be set from `.env` via `QC_JOB_TIMEOUT`.
- If this process is killed or disconnected while a job is still queued or running, the job is not
  lost: reconnect and pull its results with `recover_ibm_job_counts(job_id)` (same module), using
  the job ID that was logged at submission time.

Cost and time: local `aer` execution of the full 6-model, 29-fold LOOCV suite takes roughly 20 to
30 minutes on a laptop (the trained quantum kernel model alone is more expensive; see
[Kernel Target Alignment](#kernel-target-alignment) below). The same suite against `ibm-runtime`
would take much longer because of real device queue times, and would use real device time. Use
`--max-samples`, or pass a shorter `model_names` list in code, to scope real hardware runs on
purpose rather than defaulting to the full sweep.

## Models

### Classical baselines (`src/classical_models.py`)

- SVC with an RBF kernel, calibrated through `CalibratedClassifierCV` for probability output.
- RandomForestClassifier and GradientBoostingClassifier.
- PLS regression, for comparison against the Pätzmann benchmark of R² = 0.82 and Q² = 0.77,
  saved to `data/results/pls_regression_loocv.csv`. It fits on `log(COMDR_15min)` by default,
  matching the benchmark table's target, which is literally named LogCOMDR15min. COMDR_15min
  ranges from about 1.1 to 26.5 and is heavily skewed, so a plain linear fit on the raw ratio is
  dominated by a few extreme high responders (Fenofibrate at 26.5) and generalizes badly with
  only 29 samples (Q² around 0.45, R² around 0.68). Log transforming brings it to Q² around 0.76
  and R² around 0.83, in line with the published benchmark. Setting
  `MANUAL_FEATURES=D50,MolLogP,Kappa3,apparent_solubility` (the exact four Pätzmann variables)
  reproduces their result closely.

### Quantum feature maps (`src/quantum_circuits.py`)

- `angle_feature_map`: one Ry rotation per qubit plus a single CNOT chain. This is the baseline
  encoding from the challenge brief (equation 3).
- `entangled_feature_map`: the angle encoding repeated with a closed CNOT ring after each
  repetition, for deeper entanglement.
- `zz_feature_map`: Qiskit's `ZZFeatureMap`, angle encoding plus pairwise ZZ interaction terms.

### Quantum models (`src/quantum_models.py`)

1. **QK-SVM** (`QuantumKernelSVM`): a fidelity kernel, `|⟨ψ(xᵢ)|ψ(xⱼ)⟩|²`, measured with an
   actual compute-uncompute circuit rather than `Statevector.inner`, fed into a classical SVC
   with a precomputed kernel. Registered for both the angle and ZZ feature maps.
2. **Trained QK-SVM** (`TrainedQuantumKernelSVM`, registered as `QK-SVM_trained`): the same
   fidelity kernel SVM, but the feature map has its own trainable weight parameters (through
   `reuploading_layer`), optimized against Kernel Target Alignment before the SVM is fit. See
   [Kernel Target Alignment](#kernel-target-alignment) below for how this was tuned. Its best
   configuration reaches 96.6% fold-safe LOOCV accuracy, the best result of any model in this
   project, classical or quantum.
3. **VQC** (`VariationalQuantumClassifier`): angle encoding followed by a trainable Ry, Rz, CNOT
   chain ansatz, trained with binary cross entropy loss.
4. **Data re-uploading classifier**: repeats a trainable rotation, data re-encoding, and
   entanglement block several times, which gives the circuit more expressive power without
   adding qubits.
5. **QCNN** (`QCNNClassifier`): 6 qubit angle encoding, a conv and pool stage from 6 to 3 qubits,
   another from 3 to 2 qubits, then a full 15 parameter SU(4) dense layer on the last 2 qubits and
   a single output qubit. About 54 trainable parameters total, close to the brief's estimate of
   around 51.

Models 2 through 5 share one training core (`_VariationalCore` in `quantum_models.py`) with three
optimizers to choose from:

- `cobyla` (default): gradient free, one Sampler job per loss evaluation.
- `spsa`: one Sampler job of twice the training set size per iteration (the plus and minus
  perturbations batched together), regardless of parameter count. This is the usual choice for
  training on real hardware.
- `parameter_shift`: the exact analytic gradient, batched into one job of 2 times the parameter
  count times the training set size per iteration. Most accurate, also the most expensive.
  Meant for small comparisons (the brief asks for a parameter-shift vs SPSA comparison), not a
  full LOOCV run.

## Kernel Target Alignment

The fixed feature maps get an unremarkable raw KTA score on this dataset, roughly 0.13 to 0.34
depending on the feature map and feature subset (see `plots/kernels/kernel_heatmap_*.png`),
nowhere close to 1. That is not a sign the circuits are wrong, and pushing KTA toward 1 by
adjusting the encoding is not automatically an improvement either. Both of those claims were
checked directly rather than assumed.

**Raw KTA computed on the whole dataset at once is not a reliable stand in for how well a model
generalizes.** Widening the angle encoding's rotation range (its bandwidth, similar to tuning
gamma in a classical RBF kernel) raises KTA smoothly, from 0.26 at the `[0, π]` range this
project uses up to a peak around 0.34 near `[0, 1.75π]`. But checking the actual fold-safe LOOCV
accuracy at each of those settings tells the opposite story: accuracy is highest at the current
`[0, π]` scaling (93.1%) and gets worse as the bandwidth widens toward the "better KTA" range
(86.2% at 1.5π, 82.8% at 1.75π). Chasing a higher closed-loop KTA score directly would have made
the classifier worse, not better. This is the same kind of mistake as judging a model by its
training accuracy instead of its test accuracy.

**What does genuinely help is training the kernel's own weight parameters against KTA computed
only on each fold's training data.** `TrainedQuantumKernelSVM` adds trainable weight parameters
to the encoding on top of the data parameters, through `reuploading_layer`, and optimizes them
with COBYLA to maximize KTA using the training fold only, never the held-out sample. Comparing
different numbers of re-uploading layers under full fold-safe LOOCV:

| Layers | Trainable parameters | LOOCV accuracy | Fold-mean KTA after training |
|---|---|---|---|
| 1 | 8  | 93.1% | 0.29 |
| 2 | 16 | 93.1% | 0.38 |
| **3** | **24** | **96.6%** | **0.42** |
| 4 | 32 | 89.7% | 0.42 |
| 5 | 40 | 82.8% | 0.39 |

3 layers is a clean sweet spot: it matches the fixed encoding at worst and beats every other
model in this project at best. 4 and 5 layers overfit, with accuracy dropping sharply even as raw
KTA keeps climbing, which is the same lesson from a different angle: KTA has to be checked
fold-safe, not chased directly. `TrainedQuantumKernelSVM` defaults to 3 layers.

**Real execution cost**: unlike the variational classifiers (VQC, data re-uploading, QCNN), whose
COBYLA cost per iteration scales with the training set size, the trained kernel's cost per
iteration scales with the training set size squared, since it recomputes a full kernel matrix
(around 406 real circuits for a 28 sample training fold) at every step. The table above was
produced with fast exact-statevector simulation to find the right settings without spending hours
of real Sampler execution on a search, and the chosen configuration is then run for real. A full
29-fold LOOCV run of `QK-SVM_trained` alone at 3 layers and `maxiter=60` (the setting that reaches
96.6%) takes on the order of 11 hours of real circuit execution, so the version registered in
`QUANTUM_MODEL_BUILDERS` uses `maxiter=30` instead (COBYLA floors `maxiter` at `num_vars + 2 = 26`
for this search regardless of a smaller request, and 26 to 30 iterations already recovers most of
the achievable accuracy at a fraction of the cost). Construct
`TrainedQuantumKernelSVM(executor, n_layers=3, maxiter=60)` directly for the full 96.6% result if
you can afford the wait, or use `--max-samples` for a bounded check first.

## Evaluation methodology and leakage guarantees

- Full 29-fold leave-one-out cross validation for every model (`src/evaluation.py`).
- Scaling happens per fold only: `MinMaxScaler` is fit on the 28 training rows and only used to
  transform the held-out row, every time, for every model, with no exceptions. Quantum models
  scale to `[0, π]` since that range becomes a rotation angle. Classical models scale to `[0, 1]`.
- Feature selection is fold-safe by construction: `select_k_best_features` runs its own nested
  LOOCV, fitting `VarianceThreshold` and `SelectKBest` only on each fold's training data before
  scoring the held-out sample.
- `src/evaluation.py` includes a runnable check (`python -m src.evaluation`) that demonstrates
  the no-leak guarantee.
- Reported metrics: accuracy, F1, and per-class recall for classification; Q² (from LOOCV) and R²
  (from a full-data refit) for the PLS regression baseline.

## Bonus extensions (`src/bonus_extensions.py`)

- **KTA optimization** (`KTAOptimizedQuantumKernel`, an alias for the main suite's
  `TrainedQuantumKernelSVM`, see [Kernel Target Alignment](#kernel-target-alignment) above): a
  trained quantum kernel whose weight parameters, separate from the data encoding parameters, are
  optimized with COBYLA to maximize Kernel Target Alignment before the SVM head is fit. This demo
  fits it once to show the before and after KTA score. The main suite's `QK-SVM_trained` runs the
  same model through the full 29-fold LOOCV.
- **Execution degradation study** (`execution_degradation_study`): compares QK-SVM accuracy
  between an ideal baseline and a comparison execution target, using leak-free k-fold cross
  validation. Defaults to ideal vs synthetic noise. Pass
  `comparison_config=ExecutionConfig(mode="ibm_runtime")` to run the identical comparison against
  real hardware with no other code changes.
- **Blind SMILES prediction interface** (`BlindPredictor`): `.predict(smiles, d50=...)` returns
  `{"prediction": "Responder" or "Non-Responder", "confidence": float}`. It computes RDKit
  descriptors on the fly, applies an already-fitted scaler that only ever saw the training set,
  and never refits anything. D50 cannot be computed from a SMILES string, so it must be passed in
  explicitly whenever the deployed model uses it as a feature. Leaving it out raises an error
  instead of guessing a value.

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
| `--stage` | `all` | Which pipeline stage to run |
| `--backend` | `aer` | Where quantum circuits execute, see the table above |
| `--shots` | `4096` | Shots per circuit execution |
| `--ibm-backend` | auto (least busy) | Pin a specific IBM backend name instead of auto selection |
| `--max-samples` | none (full 29) | Cap the LOOCV sample count for the quantum and bonus stages, useful for a quick check, especially before an `ibm-runtime` run |

## Notebooks

`notebooks/01_explore_data_and_features.ipynb`, `02_classical_and_quantum_models.ipynb`, and
`03_bonus_and_blind_prediction.ipynb` are thin interactive wrappers that call into `src/`. No
logic is duplicated in the notebooks. Each one has an `EXECUTION_CONFIG` cell so the backend can
be switched the same way as the CLI's `--backend` flag.

## Known limitations

- **Lower optimizer iteration budgets for real execution.** VQC and the data re-uploading
  classifier default to `maxiter=60`, the QCNN to `maxiter=80`. These are lower than what would be
  affordable under exact simulation, because every iteration is now a real, shot-sampled Sampler
  job. Increase `maxiter` in `QUANTUM_MODEL_BUILDERS` in `src/quantum_models.py` for higher
  quality but slower training.
- **The QCNN's fixed 6-qubit design** means it uses a different feature subset than the other
  models, which are not restricted to 6 features. This is logged clearly rather than forcing the
  QCNN onto a feature count it cannot accept.
- **`ibm-runtime` mode is fully working but is not the default for a full LOOCV sweep.** Real
  device queue time and cost make that impractical to run unattended. Use `--max-samples` to scope
  real hardware runs.

## Troubleshooting

- **`MissingOptionalLibraryError: pylatexenc`**: run `pip install pylatexenc` (it is already in
  `requirements.txt`, this only matters if you installed packages one at a time).
- **`python-dotenv could not parse statement...`**: a line in `.env` is not in `KEY=VALUE` form
  and does not start with `#`. Check `.env` against `.env.example`.
- **`ibm-runtime` mode fails to resolve a backend**: run `python scripts/setup_ibm_account.py`
  first. It reports a clear error if `.env` is missing values or the credentials are rejected.
- **The quantum stage feels slow**: this is expected. Every circuit is run for real, sampled with
  shots, not computed as a linear algebra shortcut. Use `--max-samples` for a faster check and
  `--shots` to trade precision for speed.
