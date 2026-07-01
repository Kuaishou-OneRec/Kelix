#!/usr/bin/env python3
"""
Upload the Kelix README model cards (and shared assets/) to the OpenOneRec
Hugging Face repos.

For each target repo, pushes:
    1. README_Kelix-<Name>.md  -> README.md    (at repo root, overwrites default)
    2. assets/                 -> assets/      (so `<img src="assets/...">` in
                                              the README renders correctly)
    3. (optional) Deletes the legacy `assert/` folder left over from an earlier
       typo-based upload, so stale duplicates don't accumulate.

This script only uploads the README model cards and a few small figure assets
(a few hundred KB total). It is intended as a lightweight companion to
`upload_to_hf.py`, which handles the heavy bf16-sharded model checkpoints.
Run this whenever the READMEs or figures are updated.

Usage:
    # Interactive login (will prompt for a Hugging Face token):
    python examples/sana/ar_dit/opensource/upload_readmes.py

    # Or provide the token via environment variable:
    HF_TOKEN=hf_xxxxx python examples/sana/ar_dit/opensource/upload_readmes.py

    # Dry run: print what would be uploaded without touching the hub:
    DRY_RUN=1 python examples/sana/ar_dit/opensource/upload_readmes.py

    # Skip uploading assets (only push the README files):
    SKIP_ASSETS=1 python examples/sana/ar_dit/opensource/upload_readmes.py

    # Delete the legacy `assert/` folder from each repo (from an earlier
    # typo-based upload) after pushing the new `assets/` folder:
    DELETE_LEGACY_ASSERT=1 python examples/sana/ar_dit/opensource/upload_readmes.py

HF_TOKEN='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' python3 examples/sana/ar_dit/opensource/upload_readmes.py
"""

import os
import logging
from typing import Optional, List

from huggingface_hub import (
    login,
    upload_file,
    upload_folder,
    HfApi,
    create_commit,
    CommitOperationDelete,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# This file's directory (where the README_Kelix-*.md files and assets/ live).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Local folder containing figures referenced by the READMEs (via `assets/...`).
ASSETS_DIR = os.path.join(SCRIPT_DIR, "assets")

# Path-in-repo for the assets folder. Note: this is `assets`, NOT `assert` —
# the earlier `assert` spelling was a typo inherited from a sibling repo.
ASSETS_PATH_IN_REPO = "assets"

# Legacy path-in-repo left over from an earlier typo-based upload. Use
# `DELETE_LEGACY_ASSERT=1` to remove it from each repo.
LEGACY_ASSERT_PATH_IN_REPO = "assert"

# (local_readme_filename, repo_id) pairs to upload. Each README is pushed to
# the repo root as `README.md`, overwriting any default README. The shared
# `assets/` folder is also pushed to `assets/` in every target repo.
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


def _skip_assets() -> bool:
    return os.environ.get("SKIP_ASSETS", "0").strip().lower() in ("1", "true", "yes", "on")


def _delete_legacy_assert() -> bool:
    return os.environ.get("DELETE_LEGACY_ASSERT", "0").strip().lower() in ("1", "true", "yes", "on")


def _delete_legacy_assert_folder(repo_id: str) -> None:
    """Delete the legacy `assert/` folder from the repo if it still exists.

    A previous version of this script uploaded `assets/` to `assert/` (a typo).
    This removes that stale folder so it doesn't linger alongside the corrected
    `assets/` folder.
    """
    api = HfApi()
    try:
        files_in_repo: List[str] = api.list_repo_files(repo_id=repo_id, repo_type="model")
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to list files in %s: %s", repo_id, e)
        return

    legacy_files = [f for f in files_in_repo if f.startswith(LEGACY_ASSERT_PATH_IN_REPO + "/")]
    if not legacy_files:
        logger.info("No legacy `assert/` files to delete in %s", repo_id)
        return

    logger.info(
        "Deleting %d legacy `assert/` file(s) from %s: %s",
        len(legacy_files), repo_id, legacy_files,
    )
    if _is_dry_run():
        logger.info("[DRY RUN] Skipping legacy assert/ deletion in %s", repo_id)
        return

    ops = [CommitOperationDelete(path_in_repo=f) for f in legacy_files]
    create_commit(
        repo_id=repo_id,
        repo_type="model",
        operations=ops,
        commit_message="Remove legacy `assert/` folder (typo); figures now live in `assets/`",
    )
    logger.info("Deleted legacy `assert/` from %s", repo_id)


def _upload_assets(repo_id: str) -> None:
    """Upload the shared assets/ folder to `assets/` in the repo."""
    if _skip_assets():
        logger.info("SKIP_ASSETS is set; skipping assets/ upload to %s", repo_id)
        return
    if not os.path.isdir(ASSETS_DIR):
        logger.warning("No assets/ directory at %s; skipping assets upload", ASSETS_DIR)
        return

    asset_files = [f for f in os.listdir(ASSETS_DIR) if os.path.isfile(os.path.join(ASSETS_DIR, f))]
    if not asset_files:
        logger.warning("assets/ is empty; skipping assets upload to %s", repo_id)
        return

    total_kb = sum(os.path.getsize(os.path.join(ASSETS_DIR, f)) for f in asset_files) / 1024.0
    logger.info(
        "Uploading assets/ (%d file(s), %.2f KB) -> %s/ in %s",
        len(asset_files), total_kb, ASSETS_PATH_IN_REPO, repo_id,
    )

    if _is_dry_run():
        logger.info(
            "[DRY RUN] Skipping assets upload; would push %s -> %s/%s/",
            ASSETS_DIR, repo_id, ASSETS_PATH_IN_REPO,
        )
        return

    upload_folder(
        folder_path=ASSETS_DIR,
        path_in_repo=ASSETS_PATH_IN_REPO,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload README figure assets (assets/)",
    )
    logger.info("Uploaded assets/ -> %s/%s/", repo_id, ASSETS_PATH_IN_REPO)


def upload_one(readme_filename: str, repo_id: str) -> None:
    """Upload a single local README file as `README.md`, plus the shared assets/."""
    local_path = os.path.join(SCRIPT_DIR, readme_filename)
    if not os.path.isfile(local_path):
        raise FileNotFoundError(
            f"README not found: {local_path}. "
            f"Expected one of README_Kelix-DiT.md / README_Kelix-SFT.md next to this script."
        )

    size_kb = os.path.getsize(local_path) / 1024.0
    logger.info("==== %s -> %s (%.2f KB) ====", local_path, repo_id, size_kb)

    if _is_dry_run():
        logger.info("[DRY RUN] Skipping README upload; would push to %s as README.md", repo_id)
    else:
        upload_file(
            path_or_fileobj=local_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Update README model card (Kelix technique report)",
        )
        logger.info("Uploaded %s -> %s/README.md", readme_filename, repo_id)

    # Push the shared assets/ folder so the `<img src="assets/...">` paths
    # in the README render correctly on the Hub.
    _upload_assets(repo_id)

    # Optionally remove the legacy `assert/` folder from a previous typo upload.
    if _delete_legacy_assert():
        _delete_legacy_assert_folder(repo_id)


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
    if _skip_assets():
        logger.info("SKIP_ASSETS is ON: only README files will be uploaded.")
    if _delete_legacy_assert():
        logger.info("DELETE_LEGACY_ASSERT is ON: the old `assert/` folder will be removed.")

    for readme_filename, repo_id in UPLOAD_TARGETS:
        upload_one(readme_filename, repo_id)

    logger.info("All README uploads completed successfully.")


if __name__ == "__main__":
    main()
