"""Fetch SMILES strings from PubChem, with retries and rate limiting.

Looks up each drug name through the PubChem PUG-REST API. Tries the newer
ConnectivitySMILES property first and falls back to the older CanonicalSMILES
name if that fails. Waits at least 0.34 seconds between requests to stay
under PubChem's guidance of about 5 requests per second.
"""

import logging
import time

import pandas as pd
import requests
from rdkit import Chem

from src.config import DRUG_NAMES, SMILES_CSV

logger = logging.getLogger(__name__)

PUBCHEM_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/{prop}/JSON"
)
# PubChem renamed this property, so we try the new name first and the old one second.
SMILES_PROPERTIES = ["ConnectivitySMILES", "CanonicalSMILES"]
MIN_REQUEST_INTERVAL_S = 0.34
REQUEST_TIMEOUT_S = 30
MAX_RETRIES = 3


def fetch_smiles(name: str, session: requests.Session | None = None) -> str | None:
    """Return the SMILES string for a compound name, or None if not found."""
    session = session or requests
    for prop in SMILES_PROPERTIES:
        url = PUBCHEM_URL.format(name=requests.utils.quote(name), prop=prop)
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                r = session.get(url, timeout=REQUEST_TIMEOUT_S)
                if r.status_code != 200:
                    break  # property not available for this compound, try the fallback
                props = r.json()["PropertyTable"]["Properties"][0]
                for key in (prop, "ConnectivitySMILES", "CanonicalSMILES", "SMILES"):
                    if key in props and props[key]:
                        return props[key]
                break
            except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
                logger.warning("  ! %s (%s, attempt %d/%d): %s", name, prop, attempt, MAX_RETRIES, exc)
                if attempt < MAX_RETRIES:
                    time.sleep(MIN_REQUEST_INTERVAL_S)
    return None


def fetch_all_smiles(names: list[str] = DRUG_NAMES, out_path=SMILES_CSV) -> pd.DataFrame:
    """Fetch SMILES for every drug name, validate with RDKit, and save to CSV."""
    session = requests.Session()
    records = []
    for name in names:
        smiles = fetch_smiles(name, session=session)
        records.append({"drug": name, "SMILES": smiles})
        logger.info("%-24s %s", name, smiles)
        time.sleep(MIN_REQUEST_INTERVAL_S)  # stay under PubChem's rate limit

    df = pd.DataFrame(records)
    df["valid"] = df["SMILES"].apply(
        lambda s: (Chem.MolFromSmiles(s) is not None) if isinstance(s, str) else False
    )

    n_invalid = (~df["valid"]).sum()
    if n_invalid:
        invalid_names = df.loc[~df["valid"], "drug"].tolist()
        raise RuntimeError(f"{n_invalid} drug(s) failed SMILES lookup/validation: {invalid_names}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df[["drug", "SMILES"]].to_csv(out_path, index=False)
    logger.info("Saved %s (%d rows)", out_path, len(df))
    return df[["drug", "SMILES"]]


def load_or_fetch_smiles(out_path=SMILES_CSV) -> pd.DataFrame:
    """Load cached SMILES CSV if present, otherwise fetch fresh from PubChem."""
    if out_path.exists():
        logger.info("Loading cached SMILES from %s", out_path)
        return pd.read_csv(out_path)
    return fetch_all_smiles(out_path=out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    fetch_all_smiles()
