#!/usr/bin/env python3
"""Revert Muse KeyeAR checkpoint to Hugging Face checkpoint format.

风格参考：`examples/keye_ar/convert_hf_checkpoint.py`

功能：
- 输入：muse 训练/保存出来的 ckpt（state_dict）或一个包含 ckpt 的目录
- 输出：一个 Hugging Face 风格目录（包含 pytorch_model.bin 与 config.json；如提供 processor_path 也会保存 processor）

该脚本核心是调用：`KeyeARModel.revert_hf_state_dict`。

示例：

PYTHONPATH=. python3 examples/keye_ar/revert_hf_checkpoint.py \
  --muse-ckpt /path/to/muse_ckpt.pt \
  --muse-config /path/to/muse_model_dir/config.json \
  --output-dir /path/to/hf_dir \
  --processor-path /path/to/hf_processor_dir \
  --tie-word-embeddings 1

注意：
- 这里的“HF 格式”是对齐 `tests/models/keye_ar/modeling_ori.py` 所期望的 key 命名（见 `KeyeARModel.revert_hf_state_dict` 注释）。
- 如果你的 muse ckpt 是通过 `model.save_pretrained()` 存成一个目录，也可以把 `--muse-ckpt` 指向该目录。
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from transformers import AutoProcessor

from muse.config import load_config
from muse.models.keye_ar.modeling import KeyeARModel
from muse.training.common import set_default_dtype


def _maybe_load_safetensors_state_dict(model_dir: str) -> Dict[str, torch.Tensor]:
    """Load HF-style sharded safetensors or pytorch_model.bin from a directory."""
    model_dir_p = Path(model_dir)

    # Try safetensors shards first
    safetensors_files = sorted([p for p in model_dir_p.glob("*.safetensors") if "index" not in p.name])
    if safetensors_files:
        try:
            from safetensors.torch import load_file
        except ImportError as e:
            raise ImportError("Please install safetensors: pip install safetensors") from e

        sd: Dict[str, torch.Tensor] = {}
        for f in safetensors_files:
            print(f"Loading safetensors shard: {f}")
            sd.update(load_file(str(f)))
        return sd

    # Fallback pytorch_model.bin
    bin_path = model_dir_p / "pytorch_model.bin"
    if bin_path.exists():
        print(f"Loading pytorch weights: {bin_path}")
        sd = torch.load(str(bin_path), map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
            # Some checkpoints wrap state_dict
            return sd["state_dict"]
        return sd

    raise FileNotFoundError(f"No *.safetensors or pytorch_model.bin found in {model_dir}")


def _load_muse_state_dict(muse_ckpt_path: str) -> Dict[str, torch.Tensor]:
    """Load muse checkpoint state_dict.

    Supported:
    - A directory containing `pytorch_model.bin` or `*.safetensors`
    - A file path (.pt/.bin)
    """
    p = Path(muse_ckpt_path)
    if p.is_dir():
        return _maybe_load_safetensors_state_dict(str(p))

    if not p.exists():
        raise FileNotFoundError(f"muse ckpt not found: {muse_ckpt_path}")

    print(f"Loading muse ckpt file: {p}")
    sd = torch.load(str(p), map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        return sd["state_dict"]

    if not isinstance(sd, dict):
        raise ValueError(f"Unexpected checkpoint type: {type(sd)}")

    return sd


def revert_hf_checkpoint(
    muse_ckpt_path: str,
    muse_config_path: str,
    output_dir: str,
    processor_path: Optional[str] = None,
    tie_word_embeddings: bool = True,
    dtype: str = "bfloat16",
):
    """Revert muse checkpoint to HF weights and save."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load muse config
    cfg = load_config(muse_config_path)

    # Create muse model (on CPU is enough for state dict conversion)
    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]

    print(f"Building KeyeARModel from config: {muse_config_path}")
    with set_default_dtype(torch_dtype), torch.device("cpu"):
        model = KeyeARModel(cfg)

    muse_state_dict = _load_muse_state_dict(muse_ckpt_path)

    print("Reverting muse state_dict -> HF state_dict...")
    hf_state_dict = model.revert_hf_state_dict(
        muse_state_dict=muse_state_dict,
        tie_word_embeddings=tie_word_embeddings,
    )

    # Save weights
    weights_path = out_dir / "pytorch_model.bin"
    print(f"Saving HF weights to: {weights_path}")
    torch.save(hf_state_dict, str(weights_path))

    # Save config.json
    cfg_path = out_dir / "config.json"
    print(f"Saving config.json to: {cfg_path}")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2, ensure_ascii=False)

    # Save processor if provided
    if processor_path:
        print(f"Saving processor from: {processor_path}")
        processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
        processor.save_pretrained(str(out_dir))

    print("✓ Revert completed")


def main():
    parser = argparse.ArgumentParser(description="Revert Muse KeyeAR checkpoint to Hugging Face format")
    parser.add_argument(
        "--muse-ckpt",
        type=str,
        required=True,
        help="Path to muse checkpoint file (.pt/.bin) or a directory containing weights",
    )
    parser.add_argument(
        "--muse-config",
        type=str,
        required=True,
        help="Path to muse model config.json (the one used to build KeyeARModel)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory to write HF checkpoint",
    )
    parser.add_argument(
        "--processor-path",
        type=str,
        default=None,
        help="Optional: a HF processor directory to save into output-dir (tokenizer/processor files)",
    )
    parser.add_argument(
        "--tie-word-embeddings",
        type=int,
        default=1,
        help="1/0. Whether to tie word embeddings when reverting (affects lm_head.weight)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="dtype used to instantiate model on CPU for conversion",
    )

    args = parser.parse_args()

    revert_hf_checkpoint(
        muse_ckpt_path=args.muse_ckpt,
        muse_config_path=args.muse_config,
        output_dir=args.output_dir,
        processor_path=args.processor_path,
        tie_word_embeddings=bool(args.tie_word_embeddings),
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()
