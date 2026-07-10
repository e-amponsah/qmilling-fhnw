"""Central configuration: paths, constants, and column contracts shared across modules.

Every other module imports from here instead of hardcoding paths or column
names, so the pipeline can be repointed (new dataset, new output location)
by editing a single file.
"""

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
           PLOTS_DIR / "circuits", PLOTS_DIR / "kernels", PLOTS_DIR / "scatter"):
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

# Experimental particle size is an allowed feature (not leakage).
EXPERIMENTAL_FEATURE_COLS = ["D50"]

# Candidate feature pool = 14 descriptors + D50 (15 total, matches starter notebook).
CANDIDATE_FEATURE_COLS = DESCRIPTOR_COLS + EXPERIMENTAL_FEATURE_COLS

# These columns mathematically define the targets and must NEVER enter X.
LEAKAGE_COLS = ["COM_15min", "PM_15min"]

REGRESSION_TARGET = "COMDR_15min"
CLASSIFICATION_TARGET = "label_15min"

# Pätzmann et al. used D50, logD6.5 (~MolLogP), Kappa3, Sapp. Sapp (apparent
# solubility in FaSSIF) is NOT present in hackathon_dataset_15min.csv -- only
# 3 of the 4 reference variables are reproducible from the provided data.
PATZMANN_APPROX_COLS = ["D50", "MolLogP", "Kappa3"]
PATZMANN_MISSING_COLS = ["Sapp"]

RANDOM_SEED = 42
FEATURE_SELECT_K_RANGE = (4, 6)

# Pätzmann et al. 2024 benchmark (Q^2, LOOCV, COMDR_15 regression).
PATZMANN_Q2_BENCHMARK = 0.77
