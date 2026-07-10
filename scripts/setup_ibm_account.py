"""One-time setup: read IBM Cloud credentials from .env and save them as the
default account for `qiskit-ibm-runtime`, so `ExecutionConfig(mode="ibm_runtime")`
(see `src/quantum_backend.py`) can resolve a real backend without any
credentials embedded in code or passed on the command line.

Usage:
    python scripts/setup_ibm_account.py

Reads from .env (see .env.example):
    IBM_QUANTUM_CRN       -- Cloud Resource Name of your Qiskit Runtime instance
    IBM_QUANTUM_API_KEY   -- IBM Cloud API key

The account is saved to the standard qiskit-ibm-runtime credential store
(~/.qiskit/qiskit-ibm-runtime/account-config.json) under the name "default"
and set as the default account, so no further configuration is needed by
any other script in this repo.
"""

import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    env_path = root / ".env"
    if not env_path.exists():
        logger.error(
            "No .env file found at %s. Copy .env.example to .env and fill in "
            "IBM_QUANTUM_CRN and IBM_QUANTUM_API_KEY first.", env_path,
        )
        return 1
    load_dotenv(env_path)

    import os

    crn = os.environ.get("IBM_QUANTUM_CRN", "").strip()
    api_key = os.environ.get("IBM_QUANTUM_API_KEY", "").strip()
    if not crn or not api_key:
        logger.error(
            "IBM_QUANTUM_CRN and/or IBM_QUANTUM_API_KEY are missing/empty in .env. "
            "See .env.example for where to find these values."
        )
        return 1

    from qiskit_ibm_runtime import QiskitRuntimeService

    logger.info("Saving IBM Cloud account (channel='ibm_cloud')...")
    QiskitRuntimeService.save_account(
        channel="ibm_cloud",
        token=api_key,
        instance=crn,
        name="default",
        overwrite=True,
        set_as_default=True,
    )

    logger.info("Verifying: instantiating QiskitRuntimeService() and listing backends...")
    try:
        service = QiskitRuntimeService()
        backends = service.backends()
    except Exception as exc:
        logger.error("Account was saved but verification failed: %s", exc)
        return 1

    if not backends:
        logger.warning("Connected successfully, but no backends are visible on this instance.")
    else:
        logger.info("Connected. %d backend(s) available:", len(backends))
        for b in backends:
            status = b.status()
            logger.info("  - %-25s operational=%-5s pending_jobs=%d", b.name, status.operational, status.pending_jobs)

    logger.info(
        "Done. Set QC_BACKEND_MODE=ibm_runtime (in .env or the environment) to "
        "route quantum circuit execution through this account by default."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
