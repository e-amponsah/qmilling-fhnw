"""Central configuration for the pipeline: paths, constants, and column names.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

#  Paths -
ROOT_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env file.
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

# Drugs 
DRUG_NAMES = [
    "Albendazole", "Apixaban", "Apremilast", "Aripiprazole",
    "Candesartan Cilexetil", "Carbamazepine", "Celecoxib", "Ceritinib",
    "Cinnarizine", "Dasatinib", "Deferasirox", "Dipyridamole",
    "Enzalutamide", "Etoricoxib", "Ezetimibe", "Fenofibrate",
    "Gemfibrozil", "Glibenclamide", "Griseofulvin", "Indomethacin",
    "Mefenamic acid", "Nilotinib", "Nimesulide", "Olaparib",
    "Rivaroxaban", "Sertraline", "Sorafenib", "Tadalafil", "Thiabendazole",
]

# Column contracts 
# The 14 RDKit descriptors.
DESCRIPTOR_COLS = [
    "MolLogP", "MolWt", "TPSA", "NumHAcceptors", "NumHDonors",
    "NumRotatableBonds", "Kappa1", "Kappa2", "Kappa3", "Chi0v", "Chi1v",
    "BertzCT", "FractionCSP3", "RingCount",
]

# Experimental measurements that are allowed as features (not leakage):
# particle size (D50) and apparent solubility in FaSSIF.
EXPERIMENTAL_FEATURE_COLS = ["D50", "apparent_solubility"]

# Full candidate pool: 14 descriptors + D50 + apparent_solubility = 16 columns.
CANDIDATE_FEATURE_COLS = DESCRIPTOR_COLS + EXPERIMENTAL_FEATURE_COLS

# These columns mathematically define the targets and are never used as features.
LEAKAGE_COLS = ["COM_15min", "PM_15min"]

REGRESSION_TARGET = "COMDR_15min"
CLASSIFICATION_TARGET = "label_15min"

# Patzmann et al. used D50, logD6.5 (approximated here by MolLogP), Kappa3,
# and Sapp (apparent solubility in FaSSIF, provided as apparent_solubility).
PATZMANN_APPROX_COLS = ["D50", "MolLogP", "Kappa3", "apparent_solubility"]
PATZMANN_MISSING_COLS: list[str] = []

RANDOM_SEED = 42
FEATURE_SELECT_K_RANGE = (4, 6)

# Patzmann et al. 2024 benchmark: Q^2 from LOOCV regression on COMDR_15.
PATZMANN_Q2_BENCHMARK = 0.77

# Manual feature override
# Set MANUAL_FEATURES in .env as a comma separated list to skip automated
# feature selection and use an exact hand-picked list for every model.
_manual_features_raw = os.environ.get("MANUAL_FEATURES", "").strip()
MANUAL_FEATURES: list[str] | None = (
    [f.strip() for f in _manual_features_raw.split(",") if f.strip()] if _manual_features_raw else None
)

# Same idea but just for the QCNN, which always needs exactly 6 features.
# Leave unset to fall back to automated 6-feature selection.
_manual_features_qcnn_raw = os.environ.get("MANUAL_FEATURES_QCNN", "").strip()
MANUAL_FEATURES_QCNN: list[str] | None = (
    [f.strip() for f in _manual_features_qcnn_raw.split(",") if f.strip()] if _manual_features_qcnn_raw else None
)
