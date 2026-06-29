#!/usr/bin/env python3
"""
Upload released Muse checkpoints to Hugging Face Hub.

This script uploads two released model directories to the OpenOneRec Hugging Face
organization:

    1. release_dit -> https://huggingface.co/OpenOneRec/Kelix-DiT
    2. release_sft -> https://huggingface.co/OpenOneRec/Kelix-SFT

For each release directory, the script:
    1. Pushes the full model folder to the repo.
    2. Overwrites the repo-root README.md with the local README.md (if present).
    3. Pushes the local assets/ folder to assert/ in the repo (to match the
       `<img src="assert/...">` paths used inside README.md), if present.

Usage:
    # Interactive login (will prompt for a Hugging Face token):
    python examples/sana/ar_dit/opensource/upload_to_hf.py

    # Or provide the token via environment variable:
    HF_TOKEN=hf_xxxxx python examples/sana/ar_dit/opensource/upload_to_hf.py

    # Override the release root (defaults to the path in task.txt):
    RELEASE_ROOT=/path/to/release python examples/sana/ar_dit/opensource/upload_to_hf.py

HF_TOKEN='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' python3 examples/sana/ar_dit/opensource/upload_to_hf.py


"""

import os
import logging
from typing import Optional

from huggingface_hub import login, upload_folder, upload_file

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Root directory that contains both `release_dit/` and `release_sft/` sub-folders.
# Override with the `RELEASE_ROOT` environment variable if needed.
DEFAULT_RELEASE_ROOT = "/mmu_mllm_hdd_2/lingzhixin/output/release/muse"

# (local_subdir, repo_id) pairs to upload.
UPLOAD_TARGETS = [
    ("release_dit", "OpenOneRec/Kelix-DiT"),
    ("release_sft", "OpenOneRec/Kelix-SFT"),
]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("upload_to_hf")


def _resolve_release_root() -> str:
    """Return the release root, honoring the `RELEASE_ROOT` env override."""
    return os.environ.get("RELEASE_ROOT", DEFAULT_RELEASE_ROOT)


def _resolve_token() -> Optional[str]:
    """Return a Hugging Face token from env if available, else None (interactive)."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _upload_readme_and_assets(release_dir: str, repo_id: str) -> None:
    """Upload README.md and assets/ from `release_dir` to `repo_id`.

    - README.md is pushed to the repo root (overwrites any default README).
    - assets/ is pushed to assert/ in the repo, matching `img src="assert/..."`
      paths that may appear in README.md.

    Both uploads are best-effort: missing files/directories are skipped silently.
    """
    readme_path = os.path.join(release_dir, "README.md")
    if os.path.isfile(readme_path):
        logger.info("Uploading README.md -> %s", repo_id)
        upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
        )
    else:
        logger.info("No README.md found at %s, skipping", readme_path)

    assets_dir = os.path.join(release_dir, "assets")
    if os.path.isdir(assets_dir):
        logger.info("Uploading assets/ -> assert/ in %s", repo_id)
        upload_folder(
            folder_path=assets_dir,
            path_in_repo="assert",
            repo_id=repo_id,
            repo_type="model",
        )
    else:
        logger.info("No assets/ directory found at %s, skipping", assets_dir)


def upload_one(release_root: str, local_subdir: str, repo_id: str) -> None:
    """Upload a single release sub-folder to its Hugging Face repo."""
    model_path = os.path.join(release_root, local_subdir)
    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"Release directory does not exist: {model_path}. "
            f"Make sure RELEASE_ROOT is set correctly "
            f"(current: {release_root!r})."
        )

    logger.info("==== Uploading %s -> %s ====", model_path, repo_id)

    # 1. Push the model files.
    logger.info("Uploading model folder -> %s", repo_id)
    upload_folder(
        folder_path=model_path,
        repo_id=repo_id,
        repo_type="model",
    )

    # 2 & 3. Push README.md and assets/ (best-effort).
    _upload_readme_and_assets(model_path, repo_id)

    logger.info("Finished uploading %s -> %s", model_path, repo_id)


def main() -> None:
    token = _resolve_token()
    if token:
        logger.info("Logging in to Hugging Face using HF_TOKEN from environment.")
        login(token=token)
    else:
        logger.info("No HF_TOKEN in environment; launching interactive login.")
        login()

    release_root = _resolve_release_root()
    logger.info("Using release root: %s", release_root)

    for local_subdir, repo_id in UPLOAD_TARGETS:
        upload_one(release_root, local_subdir, repo_id)

    logger.info("All uploads completed successfully.")


if __name__ == "__main__":
    main()
