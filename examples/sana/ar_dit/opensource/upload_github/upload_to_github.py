#!/usr/bin/env python3
"""
Upload the Kelix open-source release to GitHub (Kuaishou-OneRec/Kelix).

This script pushes a curated subset of the repo to the public GitHub mirror at:
    https://github.com/Kuaishou-OneRec/Kelix

It uses the GitHub REST API (via `requests`) to create/overwrite files one by
one. This avoids requiring a local git clone of the target repo and works from
any machine that has network access to GitHub.

What gets uploaded (relative to the repo root of THIS repo):
  - examples/sana/ar_dit/opensource/README.md         -> README.md            (repo root)
  - examples/sana/ar_dit/opensource/assets/fig3.png   -> assets/fig3.png
  - examples/sana/ar_dit/opensource/demo_kelix.py     -> demo/demo_kelix.py
  - examples/sana/ar_dit/opensource/demo_kelix_t2i.py -> demo/demo_kelix_t2i.py
  - examples/keye_ar/train_scripts/run_train_overfit2.sh
        -> examples/keye_ar/train_scripts/run_train_overfit2.sh
  - examples/keye_ar/train_scripts/run_train_compare.json
        -> examples/keye_ar/train_scripts/run_train_compare.json
  - examples/sana/ar_dit/demo_script/debug_sft_sana.sh
        -> examples/sana/ar_dit/demo_script/debug_sft_sana.sh
  - examples/sana/ar_dit/demo_script/debug_sft.json
        -> examples/sana/ar_dit/demo_script/debug_sft.json
  - examples/sana/ar_dit/exp00debug/debug_sft.json
        -> examples/sana/ar_dit/exp00debug/debug_sft.json

The README.md at the repo root links to:
  - the arXiv technical report: https://arxiv.org/pdf/2602.09843
  - the HuggingFace model repos: OpenOneRec/Kelix-DiT and OpenOneRec/Kelix-SFT

Usage:
    # Interactive (will prompt for the GitHub token):
    python examples/sana/ar_dit/opensource/upload_github/upload_to_github.py

    # Or provide the token via environment variable:
    GITHUB_TOKEN=ghp_xxxxx python examples/sana/ar_dit/opensource/upload_github/upload_to_github.py

    # Dry run: print what would be uploaded without touching GitHub:
    DRY_RUN=1 python examples/sana/ar_dit/opensource/upload_github/upload_to_github.py

    # Override the target repo (default: Kuaishou-OneRec/Kelix):
    GITHUB_REPO=Kuaishou-OneRec/Kelix python examples/sana/ar_dit/opensource/upload_github/upload_to_github.py

    # Override the target branch (default: main):
    GITHUB_BRANCH=main python examples/sana/ar_dit/opensource/upload_github/upload_to_github.py
"""

import base64
import json
import logging
import os
import urllib.error
import urllib.request
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# This repo's root (the muse checkout). The upload paths below are resolved
# relative to this. The script lives at:
#   <repo_root>/examples/sana/ar_dit/opensource/upload_github/upload_to_github.py
# so we go up 5 levels to reach <repo_root>.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", "..", ".."))

# Target GitHub repo + branch.
GITHUB_REPO = os.environ.get("GITHUB_REPO", "Kuaishou-OneRec/Kelix")
GITHUB_BRANCH = os.environ.get("GITHUB_BRANCH", "main")

# GitHub API base.
GITHUB_API = "https://api.github.com"

# (source_path_relative_to_repo_root, path_in_github_repo) pairs to upload.
# The first entry uploads the Kelix README.md to the repo root.
UPLOAD_FILES = [
    # --- Repo root: README + figure ---
    ("examples/sana/ar_dit/opensource/upload_github/README.md", "README.md"),
    ("examples/sana/ar_dit/opensource/assets/fig3.png", "assets/fig3.png"),
    # --- Demo scripts ---
    ("examples/sana/ar_dit/opensource/demo_kelix.py", "demo/demo_kelix.py"),
    ("examples/sana/ar_dit/opensource/demo_kelix_t2i.py", "demo/demo_kelix_t2i.py"),
    # --- Example training scripts (kept in this release branch) ---
    ("examples/keye_ar/train_scripts/run_train_overfit2.sh",
     "examples/keye_ar/train_scripts/run_train_overfit2.sh"),
    ("examples/keye_ar/train_scripts/run_train_compare.json",
     "examples/keye_ar/train_scripts/run_train_compare.json"),
    ("examples/sana/ar_dit/demo_script/debug_sft_sana.sh",
     "examples/sana/ar_dit/demo_script/debug_sft_sana.sh"),
    ("examples/sana/ar_dit/demo_script/debug_sft.json",
     "examples/sana/ar_dit/demo_script/debug_sft.json"),
    ("examples/sana/ar_dit/exp00debug/debug_sft.json",
     "examples/sana/ar_dit/exp00debug/debug_sft.json"),
]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("upload_to_github")


# ---------------------------------------------------------------------------
# Env-var helpers
# ---------------------------------------------------------------------------

def _resolve_token() -> Optional[str]:
    """Return a GitHub token from env, else None (interactive prompt)."""
    return os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")


def _prompt_token() -> str:
    """Interactively prompt for a GitHub token."""
    print("No GITHUB_TOKEN in environment. Please provide a GitHub PAT.")
    print("  Create one at: https://github.com/settings/tokens (needs `repo` scope)")
    token = input("GITHUB_TOKEN: ").strip()
    if not token:
        raise SystemExit("A GitHub token is required to upload.")
    return token


def _is_dry_run() -> bool:
    return os.environ.get("DRY_RUN", "0").strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# GitHub REST API helpers (stdlib only — no `requests` dependency)
# ---------------------------------------------------------------------------

def _github_request(
    method: str,
    path: str,
    token: str,
    body: Optional[dict] = None,
    accept: str = "application/vnd.github+json",
) -> Tuple[int, dict]:
    """Make an authenticated GitHub API request. Returns (status_code, json_or_error_dict)."""
    url = f"{GITHUB_API}/repos/{GITHUB_REPO}{path}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": accept,
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "kelix-release-uploader",
    }
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            raw = resp.read().decode("utf-8")
            parsed = json.loads(raw) if raw else {}
            return resp.status, parsed
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8")
        try:
            parsed = json.loads(raw) if raw else {"message": str(e)}
        except json.JSONDecodeError:
            parsed = {"message": str(e), "body": raw}
        return e.code, parsed


def _get_file_sha(token: str, path_in_repo: str) -> Optional[str]:
    """Return the current blob SHA of a file in the GitHub repo, or None if it doesn't exist."""
    status, payload = _github_request(
        "GET",
        f"/contents/{path_in_repo}?ref={GITHUB_BRANCH}",
        token,
    )
    if status == 200 and isinstance(payload, dict) and "sha" in payload:
        return payload["sha"]
    if status == 404:
        return None
    logger.warning("GET %s returned %s: %s", path_in_repo, status, payload.get("message"))
    return None


def _create_or_update_file(
    token: str,
    path_in_repo: str,
    content_b64: str,
    message: str,
    sha: Optional[str],
) -> None:
    """Create or update a single file in the GitHub repo via the Contents API."""
    body = {
        "message": message,
        "content": content_b64,
        "branch": GITHUB_BRANCH,
    }
    if sha is not None:
        body["sha"] = sha  # update existing file

    status, payload = _github_request("PUT", f"/contents/{path_in_repo}", token, body)
    if status in (200, 201):
        action = "Updated" if sha else "Created"
        logger.info("%s %s (commit: %s)", action, path_in_repo,
                    payload.get("commit", {}).get("sha", "?")[:7])
    else:
        raise RuntimeError(
            f"Failed to upload {path_in_repo}: HTTP {status} — {payload.get('message', payload)}"
        )


# ---------------------------------------------------------------------------
# Upload logic
# ---------------------------------------------------------------------------

def _read_and_encode(local_path: str) -> str:
    """Read a local file and return its base64-encoded content (for the Contents API)."""
    with open(local_path, "rb") as f:
        raw = f.read()
    return base64.b64encode(raw).decode("ascii")


def upload_one(token: str, source_rel: str, path_in_repo: str) -> None:
    """Upload a single local file to the GitHub repo at path_in_repo."""
    local_path = os.path.join(REPO_ROOT, source_rel)
    if not os.path.isfile(local_path):
        raise FileNotFoundError(
            f"Source file not found: {local_path}. "
            f"Make sure you're running this from the muse repo root."
        )

    size_kb = os.path.getsize(local_path) / 1024.0
    logger.info("==== %s -> %s:%s (%.2f KB) ====", source_rel, GITHUB_REPO, path_in_repo, size_kb)

    if _is_dry_run():
        logger.info("[DRY RUN] Skipping upload; would push to %s:%s", GITHUB_REPO, path_in_repo)
        return

    content_b64 = _read_and_encode(local_path)
    sha = _get_file_sha(token, path_in_repo)
    action = "Update" if sha else "Create"
    _create_or_update_file(
        token,
        path_in_repo,
        content_b64,
        message=f"{action}: {path_in_repo} (Kelix open-source release)",
        sha=sha,
    )


def main() -> None:
    token = _resolve_token() or _prompt_token()

    logger.info("Target repo:   %s", GITHUB_REPO)
    logger.info("Target branch: %s", GITHUB_BRANCH)
    logger.info("Files to upload: %d", len(UPLOAD_FILES))
    if _is_dry_run():
        logger.info("DRY_RUN mode is ON: no files will be uploaded.")

    for source_rel, path_in_repo in UPLOAD_FILES:
        upload_one(token, source_rel, path_in_repo)

    logger.info("All GitHub uploads completed successfully.")
    logger.info("View at: https://github.com/%s", GITHUB_REPO)


if __name__ == "__main__":
    main()
