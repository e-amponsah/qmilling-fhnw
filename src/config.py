"""Central configuration: paths, constants, and column contracts shared across modules.

Every other module imports from here instead of hardcoding paths or column
names, so the pipeline can be repointed (new dataset, new output location)
by editing a single file.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# --- Paths -------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent

# Load .env (IBM_QUANTUM_CRN, IBM_QUANTUM_API_KEY, QC_BACKEND_MODE, ...) once,
# here, so every module that imports src.config picks up the same environment
# without needing its own load_dotenv() call. Silently a no-op if .env is absent.
load_dotenv(ROOT_DIR / ".env")
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"
PLOTS_DIR = ROOT_DIR / "plots"

for _d in (RAW_DIR, PROCESSED_DIR, RESULTS_DIR, PLOTS_DIR,
           PLOTS_DIR / "circuits", PLOTS_DIR / "kernels", PLOTS_DIR / "scatter", PLOTS_DIR / "pca"):
    _d.mkdir(parents=True, exist_ok=True)

SMILES_CSV = RAW_DIR / "smiles_29_drugs.csv"
DESCRIPTORS_CSV = PROCESSED_DIR / "rdkit_descriptors_29_drugs.csv"
DATASET_CSV = RAW_DIR / "hackathon_dataset_15min.csv"
MODELING_TABLE_CSV = PROCESSED_DIR / "modeling_table.csv"

# --- Drugs ---------------------------------------------------------------
DRUG_NAMES = [
    "Albendazole", "Apixaban", "Apremilast", "Aripiprazole",
    "Candesartan Cilexetil", "Carbamazepine", "Celecoxib", "Ceritinib",
    "Cinnarizine", "Dasatinib", "Deferasirox", "Dipyridamole",
    "Enzalutamide", "Etoricoxib", "Ezetimibe", "Fenofibrate",
    "Gemfibrozil", "Glibenclamide", "Griseofulvin", "Indomethacin",
    "Mefenamic acid", "Nilotinib", "Nimesulide", "Olaparib",
    "Rivaroxaban", "Sertraline", "Sorafenib", "Tadalafil", "Thiabendazole",
]

# --- Column contracts ------------------------------------------------------
# The 14 RDKit descriptors specified by the challenge starter code.
DESCRIPTOR_COLS = [
    "MolLogP", "MolWt", "TPSA", "NumHAcceptors", "NumHDonors",
    "NumRotatableBonds", "Kappa1", "Kappa2", "Kappa3", "Chi0v", "Chi1v",
    "BertzCT", "FractionCSP3", "RingCount",
]

# Experimental columns that are allowed features (not leakage): particle
# size (D50) and apparent solubility in FaSSIF (Sapp).
EXPERIMENTAL_FEATURE_COLS = ["D50", "apparent_solubility"]

# Candidate feature pool = 14 descriptors + D50 + apparent_solubility (16 total).
CANDIDATE_FEATURE_COLS = DESCRIPTOR_COLS + EXPERIMENTAL_FEATURE_COLS

# These columns mathematically define the targets and must NEVER enter X.
LEAKAGE_COLS = ["COM_15min", "PM_15min"]

REGRESSION_TARGET = "COMDR_15min"
CLASSIFICATION_TARGET = "label_15min"

# Pätzmann et al. used D50, logD6.5 (~MolLogP), Kappa3, Sapp (apparent
# solubility in FaSSIF, provided here as `apparent_solubility`). All four
# reference variables are reproducible from the provided data.
PATZMANN_APPROX_COLS = ["D50", "MolLogP", "Kappa3", "apparent_solubility"]
PATZMANN_MISSING_COLS: list[str] = []

RANDOM_SEED = 42
FEATURE_SELECT_K_RANGE = (4, 6)

# Pätzmann et al. 2024 benchmark (Q^2, LOOCV, COMDR_15 regression).
PATZMANN_Q2_BENCHMARK = 0.77

# --- Manual feature override ------------------------------------------------
# Set MANUAL_FEATURES in .env (comma-separated column names, e.g.
# "D50,MolLogP,Kappa3,TPSA") to skip the automated SelectKBest sweep and use
# an exact, hand-picked feature list for every model instead -- e.g. after
# eyeballing plots/feature_correlation_heatmap.png. Leave unset/empty to keep
# the automated fold-safe selection in src/features.py::select_k_best_features.
# Names are validated against CANDIDATE_FEATURE_COLS at use time (see
# src/features.py::get_modeling_features), not here, to keep this a plain
# constant with no cross-module import.
_manual_features_raw = os.environ.get("MANUAL_FEATURES", "").strip()
MANUAL_FEATURES: list[str] | None = (
    [f.strip() for f in _manual_features_raw.split(",") if f.strip()] if _manual_features_raw else None
)
