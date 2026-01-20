"""KeyeAR ckpt round-trip revert 验证脚本（HF -> muse -> HF）。

目标：
1) 从 HF converted ckpt 目录加载原始 HF(state_dict)
2) 按 `load_keye_ar_model_v2` 的逻辑把它 convert 成 muse(KeyeARModel) 的 state_dict 并 load 进模型
3) 再用 `KeyeARModel.revert_hf_state_dict` 把 muse 的 state_dict 还原回 HF 格式
4) 对比 "原始 HF" vs "round-trip 后 HF" 是否完全一致

注意：这是 demo/脚本风格（非 pytest），直接运行：

    python tests/models/keye_ar/test_keye_ar_revert.py

你可以在 main 里修改 `HF_CKPT_DIR`。
"""

from __future__ import annotations

import os
import json
from typing import Dict, Tuple

import torch
from safetensors.torch import load_file
from transformers import AutoProcessor

from muse.models.keye_ar.modeling import KeyeARModel
from tests.models.keye_ar.test_verify_logits_consistency_v2_clean import load_keye_ar_config


DTYPE = torch.bfloat16
DEVICE = torch.device("cuda:0")


def _require_cuda_bf16() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用：该脚本默认要求 GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前 GPU/驱动不支持 bfloat16：该脚本默认要求 bf16")


def _load_hf_state_dict_from_dir(output_model_dir: str) -> Dict[str, torch.Tensor]:
    state_dict: Dict[str, torch.Tensor] = {}
    for fn in os.listdir(output_model_dir):
        if fn.endswith(".safetensors"):
            print(f"[load] {fn}")
            state_dict.update(load_file(os.path.join(output_model_dir, fn)))
    print(f"[load] total hf keys={len(state_dict)}")
    return state_dict


def _cast_state_dict_to_cpu_fp32(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """比较用：全部转 cpu+fp32（避免 bf16 的精度/设备差异干扰）。"""
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if torch.is_tensor(v):
            out[k] = v.detach().cpu().float()
    return out


def _compare_state_dicts(
    a: Dict[str, torch.Tensor],
    b: Dict[str, torch.Tensor],
    *,
    name_a: str,
    name_b: str,
    rtol: float = 0.0,
    atol: float = 0.0,
    print_topk: int = 50,
) -> None:
    keys_a = set(a.keys())
    keys_b = set(b.keys())

    missing = sorted(keys_a - keys_b)
    extra = sorted(keys_b - keys_a)

    print(f"[compare] {name_a} keys={len(keys_a)}, {name_b} keys={len(keys_b)}")
    print(f"[compare] missing_in_{name_b}={len(missing)}, extra_in_{name_b}={len(extra)}")

    if missing:
        print(f"  missing (top {min(print_topk, len(missing))}):")
        for k in missing[:print_topk]:
            print("   -", k)

    for k in list(missing):
        if k.startswith("visual_tokenizer.visual.vision_model.head."):
            print(f"key {k} is allowed to be missing")
            del missing[missing.index(k)]

    if extra:
        print(f"  extra (top {min(print_topk, len(extra))}):")
        for k in extra[:print_topk]:
            print("   +", k)

    assert not missing and not extra, "state_dict keys mismatch"

    max_diff = 0.0
    max_diff_k = None
    max_shape_k = None

    for k in keys_a:
        ta = a[k]
        tb = b[k]

        if ta.shape != tb.shape:
            max_shape_k = k
            print(f"[shape mismatch] key={k}, {name_a}={tuple(ta.shape)} vs {name_b}={tuple(tb.shape)}")
            raise AssertionError("state_dict tensor shape mismatch")

        if not torch.allclose(ta, tb, rtol=rtol, atol=atol):
            diff = (ta - tb).abs().max().item()
            if diff > max_diff:
                max_diff = diff
                max_diff_k = k

    print(f"[compare] max_diff={max_diff}, max_diff_key={max_diff_k}, max_shape_mismatch_key={max_shape_k}")
    assert max_diff == 0.0, f"tensor mismatch max_diff={max_diff} key={max_diff_k}"


def load_keye_ar_model_v2_for_revert(output_model_dir: str) -> Tuple[KeyeARModel, Dict[str, torch.Tensor]]:
    """参考 `test_verify_logits_consistency_v2_clean.load_keye_ar_model_v2` 的加载方式。

    返回：
    - model: 已 load muse(converted) 权重的 KeyeARModel
    - hf_state_dict: 原始 HF 格式权重（用于最后对比）
    """

    processor = AutoProcessor.from_pretrained(output_model_dir, trust_remote_code=True)
    _ = processor

    config = load_keye_ar_config(f"{output_model_dir}/config.json")
    model = KeyeARModel(config)

    hf_state_dict = _load_hf_state_dict_from_dir(output_model_dir)

    converted_state_dict = model.convert_hf_state_dict(hf_state_dict, tie_word_embeddings=False)
    print(f"[convert] to muse keys={len(converted_state_dict)}")

    missing, unexpected = model.load_state_dict(converted_state_dict, strict=True)
    assert len(missing) == 0 and len(unexpected) == 0

    model = model.to(device=DEVICE, dtype=DTYPE)
    model.eval()

    return model, hf_state_dict


def demo_round_trip_revert(hf_ckpt_dir: str) -> None:
    print("\n==================== demo_round_trip_revert ====================")
    print(f"hf_ckpt_dir={hf_ckpt_dir}")

    model, hf_state_dict = load_keye_ar_model_v2_for_revert(hf_ckpt_dir)

    # 导出 muse state_dict（注意：这里拿的是模型自身的参数命名）
    muse_sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    print(f"[muse] model.state_dict keys={len(muse_sd)}")

    # muse -> hf
    reverted_hf_sd = model.revert_hf_state_dict(muse_sd, tie_word_embeddings=False)
    print(f"[revert] reverted hf keys={len(reverted_hf_sd)}")

    # 对比（用 cpu+fp32 做严格相等）
    hf_a = _cast_state_dict_to_cpu_fp32(hf_state_dict)
    hf_b = _cast_state_dict_to_cpu_fp32(reverted_hf_sd)

    _compare_state_dicts(hf_a, hf_b, name_a="hf_original", name_b="hf_roundtrip", rtol=0.0, atol=0.0)
    print("[ok] round-trip revert success: hf_original == hf_roundtrip")


if __name__ == "__main__":
    os.environ["nosp"] = "true"
    os.environ["Qwen3RMSNorm_fp32"] = "0"

    _require_cuda_bf16()

    HF_CKPT_DIR = "/mmu_mllm_hdd_2/zhouyang12/output/Keye/vqar_11.7/run_8b_vis_stage3.29_1e-4/step4000/global_step4000/converted"

    demo_round_trip_revert(HF_CKPT_DIR)