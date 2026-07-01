#!/usr/bin/env python3
"""
Upload the Kelix README model cards to the OpenOneRec Hugging Face repos.

Pushes:
    README_Kelix-DiT.md -> OpenOneRec/Kelix-DiT   (as README.md at repo root)
    README_Kelix-SFT.md -> OpenOneRec/Kelix-SFT   (as README.md at repo root)

This script only uploads the README model cards (a few KB each). It is intended
to be a lightweight companion to `upload_to_hf.py`, which handles the heavy
bf16-sharded model checkpoints. Run this whenever the READMEs are updated.

Usage:
    # Interactive login (will prompt for a Hugging Face token):
    python examples/sana/ar_dit/opensource/upload_readmes.py

    # Or provide the token via environment variable:
    HF_TOKEN=hf_xxxxx python examples/sana/ar_dit/opensource/upload_readmes.py

    # Dry run: print what would be uploaded without touching the hub:
    DRY_RUN=1 python examples/sana/ar_dit/opensource/upload_readmes.py

HF_TOKEN='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' python3 examples/sana/ar_dit/opensource/upload_readmes.py
"""

import os
import logging
from typing import Optional

from huggingface_hub import login, upload_file

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# This file's directory (where the README_Kelix-*.md files live).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# (local_readme_filename, repo_id) pairs to upload. Each README is pushed to
# the repo root as `README.md`, overwriting any default README.
UPLOAD_TARGETS = [
    ("README_Kelix-DiT.md", "OpenOneRec/Kelix-DiT"),
    ("README_Kelix-SFT.md", "OpenOneRec/Kelix-SFT"),
]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("upload_readmes")


def _resolve_token() -> Optional[str]:
    """Return a Hugging Face token from env if available, else None (interactive)."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _is_dry_run() -> bool:
    return os.environ.get("DRY_RUN", "0").strip().lower() in ("1", "true", "yes", "on")


def upload_one(readme_filename: str, repo_id: str) -> None:
    """Upload a single local README file to a Hugging Face repo as `README.md`."""
    local_path = os.path.join(SCRIPT_DIR, readme_filename)
    if not os.path.isfile(local_path):
        raise FileNotFoundError(
            f"README not found: {local_path}. "
            f"Expected one of README_Kelix-DiT.md / README_Kelix-SFT.md next to this script."
        )

    size_kb = os.path.getsize(local_path) / 1024.0
    logger.info("==== %s -> %s (%.2f KB) ====", local_path, repo_id, size_kb)

    if _is_dry_run():
        logger.info("[DRY RUN] Skipping upload; would push to %s as README.md", repo_id)
        return

    upload_file(
        path_or_fileobj=local_path,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Update README model card (Kelix technique report)",
    )
    logger.info("Uploaded %s -> %s/README.md", readme_filename, repo_id)


def main() -> None:
    token = _resolve_token()
    if token:
        logger.info("Logging in to Hugging Face using HF_TOKEN from environment.")
        login(token=token)
    else:
        logger.info("No HF_TOKEN in environment; launching interactive login.")
        login()

    if _is_dry_run():
        logger.info("DRY_RUN mode is ON: no files will be uploaded.")

    for readme_filename, repo_id in UPLOAD_TARGETS:
        upload_one(readme_filename, repo_id)

    logger.info("All README uploads completed successfully.")


if __name__ == "__main__":
    main()
