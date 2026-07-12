"""Minimal real-hardware demo: train VQC locally (aer_simulator), then run
ONLY inference for a handful of held-out drugs on real IBM Quantum
hardware.

This intentionally skips a full LOOCV run on real hardware. VQC's COBYLA
training loop needs one job per iteration (maxiter=60), so training on
real hardware would mean dozens of sequential queued jobs -- slow and
expensive for what this script is for: producing a genuine IBM Runtime
job_id + backend name as proof that the pipeline runs on real hardware,
alongside a few real predictions to sanity-check against ground truth.
Inference itself is a single circuit per sample, so this is one job total.

Usage: PYTHONPATH=. python scripts/ibm_cloud_vqc_demo.py [--n-holdout 5]
"""
import argparse
import json
import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import CLASSIFICATION_TARGET, DATASET_CSV, DESCRIPTORS_CSV, RANDOM_SEED, RESULTS_DIR
from src.features import build_descriptor_table, build_modeling_table, get_candidate_matrix, get_modeling_features
from src.data_fetching import load_or_fetch_smiles
from src.quantum_backend import ExecutionConfig, QuantumExecutor
from src.quantum_models import VariationalQuantumClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Real-hardware VQC inference demo")
    parser.add_argument("--n-holdout", type=int, default=5, help="Number of drugs to predict on real IBM hardware")
    args = parser.parse_args()

    smiles_df = load_or_fetch_smiles()
    descriptors = pd.read_csv(DESCRIPTORS_CSV) if DESCRIPTORS_CSV.exists() else build_descriptor_table(smiles_df)
    data = build_modeling_table(descriptors, dataset_path=DATASET_CSV)
    X_full = get_candidate_matrix(data)
    y = data[CLASSIFICATION_TARGET].values
    selection = get_modeling_features(X_full, y)
    selected_features = selection["selected_features"]
    X = data[selected_features].values
    drugs = data["drug"].values

    X_train, X_test, y_train, y_test, drugs_train, drugs_test = train_test_split(
        X, y, drugs, test_size=args.n_holdout, random_state=RANDOM_SEED, stratify=y
    )
    logger.info("Train on %d drugs, hold out %d for real-hardware inference: %s", len(X_train), len(X_test), list(drugs_test))

    logger.info("Training VQC locally on aer_simulator...")
    aer_executor = QuantumExecutor(ExecutionConfig(mode="aer_simulator", shots=4096))
    model = VariationalQuantumClassifier(executor=aer_executor, n_layers=2, optimizer="cobyla", maxiter=60, random_state=RANDOM_SEED + 2)
    model.fit(X_train, y_train)
    logger.info("Local training done. Switching to real IBM Quantum hardware for inference only...")

    ibm_executor = QuantumExecutor(ExecutionConfig(mode="ibm_runtime", shots=4096))
    model.executor = ibm_executor

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    logger.info("Real-hardware backend: %s", ibm_executor.config.label())
    for drug, true_y, pred_y, p in zip(drugs_test, y_test, preds, proba):
        logger.info("  %-20s true=%d pred=%d P(responder)=%.3f", drug, true_y, pred_y, p)
    accuracy = float((preds == y_test).mean())
    logger.info("Held-out accuracy on real hardware: %.3f (%d/%d)", accuracy, int((preds == y_test).sum()), len(y_test))

    out_path = RESULTS_DIR / "ibm_cloud_vqc_demo.json"
    payload = {
        "backend": ibm_executor.backend.name,
        "shots": ibm_executor.config.shots,
        "drugs": list(drugs_test),
        "y_true": [int(v) for v in y_test],
        "y_pred": [int(v) for v in preds],
        "p_responder": [float(v) for v in proba],
        "accuracy": accuracy,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Saved %s", out_path)


if __name__ == "__main__":
    main()
