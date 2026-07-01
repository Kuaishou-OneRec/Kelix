#!/usr/bin/env python3
"""
Upload released Muse checkpoints to Hugging Face Hub.

This script uploads two released model directories to the OpenOneRec Hugging Face
organization:

    1. release_dit -> https://huggingface.co/OpenOneRec/Kelix-DiT
    2. release_sft -> https://huggingface.co/OpenOneRec/Kelix-SFT

For each release directory, the script:
    1. (Optional) Deletes previously uploaded `*.safetensors` and
       `model.safetensors.index.json` from the target repo so stale shards
       don't accumulate.
    2. Creates a temporary directory that mirrors the original model directory,
       but with every `*.safetensors` converted to bfloat16 and split into
       ~`MAX_GB_PER_SHARD` GiB parts named `model-XXXXX-of-YYYYY.safetensors`
       together with a regenerated `model.safetensors.index.json`.
       Non-safetensors files (config.json, README.md, ...) are copied as-is,
       and the local `assets/` folder is copied to `assets/` to match the
       `<img src="assets/...">` paths used inside README.md.
    3. Pushes the temporary folder to the repo.
    4. Removes the temporary directory.

Usage:
    # Interactive login (will prompt for a Hugging Face token):
    python examples/sana/ar_dit/opensource/upload_to_hf.py

    # Or provide the token via environment variable:
    HF_TOKEN=hf_xxxxx python examples/sana/ar_dit/opensource/upload_to_hf.py

    # Override the release root (defaults to the path in task.txt):
    RELEASE_ROOT=/path/to/release python examples/sana/ar_dit/opensource/upload_to_hf.py

    # Delete previously uploaded safetensors before re-upload:
    DELETE_PREVIOUS_SAFETENSORS=1 python examples/sana/ar_dit/opensource/upload_to_hf.py

    # Override the per-shard size (in GiB, default 4):
    MAX_GB_PER_SHARD=2 python examples/sana/ar_dit/opensource/upload_to_hf.py

    # Override where the temporary directory is created (defaults to the
    # parent of the release folder, so it shares the same disk):
    TEMP_DIR_ROOT=/scratch/tmp python examples/sana/ar_dit/opensource/upload_to_hf.py

HF_TOKEN='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' python3 examples/sana/ar_dit/opensource/upload_to_hf.py


"""

import os
import json
import shutil
import logging
import tempfile
from typing import Optional, Dict, List

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from huggingface_hub import (
    login,
    upload_folder,
    HfApi,
    create_commit,
    CommitOperationDelete,
)

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

# Target size of each safetensors shard, in GiB.
DEFAULT_MAX_GB_PER_SHARD = 4

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("upload_to_hf")


# ---------------------------------------------------------------------------
# Env-var resolution helpers
# ---------------------------------------------------------------------------

def _resolve_release_root() -> str:
    """Return the release root, honoring the `RELEASE_ROOT` env override."""
    return os.environ.get("RELEASE_ROOT", DEFAULT_RELEASE_ROOT)


def _resolve_token() -> Optional[str]:
    """Return a Hugging Face token from env if available, else None (interactive)."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _resolve_max_gb_per_shard() -> int:
    raw = os.environ.get("MAX_GB_PER_SHARD")
    if not raw:
        return DEFAULT_MAX_GB_PER_SHARD
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            "Invalid MAX_GB_PER_SHARD=%r, using default %d",
            raw, DEFAULT_MAX_GB_PER_SHARD,
        )
        return DEFAULT_MAX_GB_PER_SHARD


def _resolve_delete_previous() -> bool:
    raw = os.environ.get("DELETE_PREVIOUS_SAFETENSORS", "0")
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _resolve_temp_dir_root(source_dir: str) -> str:
    """Return the parent dir for the temporary staging directory.

    Defaults to the parent of `source_dir` so the staging dir lives on the
    same filesystem (and therefore has enough free space for a bf16 copy of
    the model). Override with the `TEMP_DIR_ROOT` env var.
    """
    return os.environ.get("TEMP_DIR_ROOT") or os.path.dirname(os.path.abspath(source_dir))


# ---------------------------------------------------------------------------
# Hugging Face repo helpers
# ---------------------------------------------------------------------------

def _delete_previous_safetensors(repo_id: str) -> None:
    """Delete all `*.safetensors` and `model.safetensors.index.json` from the repo.

    Uses a single `create_commit` with one `CommitOperationDelete` per file so
    the deletion is atomic.
    """
    api = HfApi()
    try:
        files_in_repo: List[str] = api.list_repo_files(repo_id=repo_id, repo_type="model")
    except Exception as e:  # noqa: BLE001 - we want to keep going if listing fails
        logger.warning("Failed to list files in %s: %s", repo_id, e)
        return

    to_delete = [
        f for f in files_in_repo
        if f.endswith(".safetensors") or f == "model.safetensors.index.json"
    ]
    if not to_delete:
        logger.info("No previous safetensors to delete in %s", repo_id)
        return

    logger.info(
        "Deleting %d previous safetensors file(s) from %s: %s",
        len(to_delete), repo_id, to_delete,
    )
    ops = [CommitOperationDelete(path_in_repo=f) for f in to_delete]
    create_commit(
        repo_id=repo_id,
        repo_type="model",
        operations=ops,
        commit_message="Delete previous safetensors before re-upload (bf16 + sharded)",
    )
    logger.info("Deleted previous safetensors from %s", repo_id)


# ---------------------------------------------------------------------------
# Safetensors -> bf16 sharding
# ---------------------------------------------------------------------------

def _shard_filename(shard_idx: int, num_shards: int, placeholder: bool = False) -> str:
    """Return the standard `model-XXXXX-of-YYYYY.safetensors` filename.

    When `placeholder=True`, the `num_shards` part is `XXXXX` so we can save
    shards before knowing the final count; they are renamed afterwards.
    """
    if placeholder:
        return f"model-{str(shard_idx).zfill(5)}-of-XXXXX.safetensors"
    return f"model-{str(shard_idx).zfill(5)}-of-{str(num_shards).zfill(5)}.safetensors"


def _process_safetensors(
    source_dir: str,
    temp_dir: str,
    max_gb_per_shard: int,
) -> int:
    """Convert every `*.safetensors` in `source_dir` to bf16 and shard into
    ~`max_gb_per_shard` GiB parts written into `temp_dir`.

    Also writes a `model.safetensors.index.json` mapping every tensor key to
    its shard. Returns the number of shards written.
    """
    safetensors_files = sorted(
        f for f in os.listdir(source_dir) if f.endswith(".safetensors")
    )
    if not safetensors_files:
        logger.info("No safetensors files found in %s, skipping sharding", source_dir)
        return 0

    max_bytes = max_gb_per_shard * 1024 * 1024 * 1024
    logger.info(
        "Sharding %d safetensors file(s) from %s into ~%d GiB bf16 parts",
        len(safetensors_files), source_dir, max_gb_per_shard,
    )

    current_shard: Dict[str, torch.Tensor] = {}
    current_size = 0
    shard_idx = 0
    weight_map: Dict[str, int] = {}  # tensor_key -> shard_idx
    total_size = 0
    shard_paths: Dict[int, str] = {}  # shard_idx -> placeholder path

    def _flush() -> None:
        nonlocal current_shard, current_size, shard_idx
        if not current_shard:
            return
        shard_path = os.path.join(temp_dir, _shard_filename(shard_idx, 0, placeholder=True))
        save_file(current_shard, shard_path, metadata={"format": "pt"})
        logger.info(
            "Saved shard %d (%.2f GiB, %d tensors)",
            shard_idx, os.path.getsize(shard_path) / 1024**3, len(current_shard),
        )
        shard_paths[shard_idx] = shard_path
        current_shard = {}
        current_size = 0
        shard_idx += 1

    for st_file in safetensors_files:
        st_path = os.path.join(source_dir, st_file)
        logger.info("Loading %s ...", st_path)
        with safe_open(st_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                if tensor.dtype != torch.bfloat16:
                    tensor = tensor.to(torch.bfloat16)
                tensor_size = tensor.numel() * tensor.element_size()
                current_shard[key] = tensor
                current_size += tensor_size
                total_size += tensor_size
                weight_map[key] = shard_idx
                if current_size >= max_bytes:
                    _flush()

    _flush()  # write the trailing shard

    num_shards = shard_idx
    if num_shards == 0:
        logger.warning("No tensors found in any safetensors file in %s", source_dir)
        return 0

    # Rename placeholder shards to their final `model-XXXXX-of-YYYYY` names and
    # build the final weight_map (tensor_key -> shard filename).
    final_weight_map: Dict[str, str] = {}
    for s_idx, s_path in shard_paths.items():
        final_name = _shard_filename(s_idx, num_shards, placeholder=False)
        final_path = os.path.join(temp_dir, final_name)
        os.rename(s_path, final_path)
        for key, k_shard_idx in weight_map.items():
            if k_shard_idx == s_idx:
                final_weight_map[key] = final_name

    index_path = os.path.join(temp_dir, "model.safetensors.index.json")
    with open(index_path, "w") as f:
        json.dump(
            {
                "metadata": {"total_size": total_size},
                "weight_map": final_weight_map,
            },
            f,
            indent=2,
        )
    logger.info(
        "Wrote %d shard(s) and model.safetensors.index.json (total bf16 size: %.2f GiB)",
        num_shards, total_size / 1024**3,
    )
    return num_shards


# ---------------------------------------------------------------------------
# Staging helpers
# ---------------------------------------------------------------------------

def _copy_non_safetensors(source_dir: str, temp_dir: str) -> None:
    """Copy everything except `*.safetensors` and `model.safetensors.index.json`
    from `source_dir` into `temp_dir`.

    The local `assets/` subdirectory (if present) is copied to `assets/` inside
    `temp_dir` to match the `<img src="assets/...">` paths used in README.md.
    """
    for entry in os.listdir(source_dir):
        src_path = os.path.join(source_dir, entry)
        if entry.endswith(".safetensors"):
            continue  # regenerated by _process_safetensors
        if entry == "model.safetensors.index.json":
            continue  # regenerated by _process_safetensors
        if os.path.isdir(src_path) and entry == "assets":
            dst_path = os.path.join(temp_dir, "assets")
            logger.info("Copying assets/ -> assets/")
            shutil.copytree(src_path, dst_path)
            continue
        if os.path.isdir(src_path):
            shutil.copytree(src_path, os.path.join(temp_dir, entry))
        else:
            shutil.copy2(src_path, os.path.join(temp_dir, entry))


def upload_one(
    release_root: str,
    local_subdir: str,
    repo_id: str,
    max_gb_per_shard: int,
    delete_previous: bool,
) -> None:
    """Stage and upload a single release sub-folder to its Hugging Face repo."""
    model_path = os.path.join(release_root, local_subdir)
    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"Release directory does not exist: {model_path}. "
            f"Make sure RELEASE_ROOT is set correctly "
            f"(current: {release_root!r})."
        )

    logger.info("==== Processing %s -> %s ====", model_path, repo_id)

    if delete_previous:
        _delete_previous_safetensors(repo_id)

    # Create the staging directory next to the source so it shares the same
    # disk (and therefore has room for a bf16 copy of the model).
    temp_parent = _resolve_temp_dir_root(model_path)
    os.makedirs(temp_parent, exist_ok=True)
    temp_dir = tempfile.mkdtemp(prefix=f"upload_hf_{local_subdir}_", dir=temp_parent)
    logger.info("Created temporary staging directory: %s", temp_dir)

    try:
        # 1. Copy non-safetensors files (config.json, README.md, assets -> assets, ...).
        _copy_non_safetensors(model_path, temp_dir)

        # 2. Convert safetensors to bf16 and shard into ~max_gb_per_shard parts.
        num_shards = _process_safetensors(model_path, temp_dir, max_gb_per_shard)
        if num_shards == 0:
            logger.warning("No safetensors were written for %s", model_path)

        # 3. Upload the staged folder (repo root == temp_dir contents).
        logger.info("Uploading staged folder -> %s", repo_id)
        upload_folder(
            folder_path=temp_dir,
            repo_id=repo_id,
            repo_type="model",
        )

        logger.info("Finished uploading %s -> %s", model_path, repo_id)
    finally:
        logger.info("Cleaning up temporary staging directory: %s", temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> None:
    token = _resolve_token()
    if token:
        logger.info("Logging in to Hugging Face using HF_TOKEN from environment.")
        login(token=token)
    else:
        logger.info("No HF_TOKEN in environment; launching interactive login.")
        login()

    release_root = _resolve_release_root()
    max_gb_per_shard = _resolve_max_gb_per_shard()
    delete_previous = _resolve_delete_previous()

    logger.info("Using release root: %s", release_root)
    logger.info("Max GiB per shard: %d", max_gb_per_shard)
    logger.info("Delete previous safetensors before upload: %s", delete_previous)

    for local_subdir, repo_id in UPLOAD_TARGETS:
        upload_one(
            release_root=release_root,
            local_subdir=local_subdir,
            repo_id=repo_id,
            max_gb_per_shard=max_gb_per_shard,
            delete_previous=delete_previous,
        )

    logger.info("All uploads completed successfully.")


if __name__ == "__main__":
    main()
