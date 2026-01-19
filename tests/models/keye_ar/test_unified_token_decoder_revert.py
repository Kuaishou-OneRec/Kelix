"""UnifiedTokenDecoder demo + 权重转换/回滚验证（GPU + bf16）。

注意：你要的是 demo 脚本风格（非 pytest）。
- 直接 `python tests/models/keye_ar/test_unified_token_decoder_revert.py` 即可。
- 入口在 `__main__`，会依次调用各个 demo，并打印调试信息。

约束：
- 不允许 CPU 跑；全部使用 `cuda + torch.bfloat16`。
"""

from __future__ import annotations

import copy

import torch

from muse.config.model_config import UnifiedTokenDecoderConfig
from muse.models.keye_ar.unified_token_decoder import UnifiedTokenDecoder
from tests.models.keye_ar.original_transformer_decoder import PureDecoderTransformer


DTYPE = torch.bfloat16
DEVICE = torch.device("cuda")


def _require_cuda_bf16() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用：该 demo 强制要求 GPU 运行")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前 GPU/驱动不支持 bfloat16：该 demo 强制要求 torch.bfloat16")


def _tensor_stat(x: torch.Tensor) -> str:
    xf = x.detach().float()
    return (
        f"shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device}, "
        f"min={xf.min().item():.6g}, max={xf.max().item():.6g}, mean={xf.mean().item():.6g}"
    )


def _to_cuda_bf16(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if torch.is_tensor(v):
            out[k] = v.detach().to(device=DEVICE, dtype=DTYPE)
    return out


def _make_models(
    *,
    vocab_size: int = 128,
    max_length: int = 64,
    max_pos_length: int | None = None,
    d_model: int = 64,
    nhead: int = 8,
    num_layers: int = 2,
    dim_feedforward: int = 256,
    reduce: bool = False,
):
    if max_pos_length is None:
        max_pos_length = max_length

    cfg = UnifiedTokenDecoderConfig(
        vocab_size=vocab_size,
        max_length=max_length,
        max_pos_length=max_pos_length,
        d_model=d_model,
        eos_token=0,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        use_gradient_checkpointing=False,
        input_dim=d_model if reduce else None,
        reduce=reduce,
        attention_function="flash_attention_2",
    )

    # 让 ref/uni 初始化一致，方便 debug
    torch.manual_seed(0)
    ref = PureDecoderTransformer(
        vocab_size=vocab_size,
        max_length=max_length,
        d_model=d_model,
        eos_token=0,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        use_flash_attn=False,
        use_gradient_checkpointing=False,
        input_dim=d_model if reduce else None,
        reduce=reduce,
        lm_head=None,
    ).to(device=DEVICE, dtype=DTYPE)

    torch.manual_seed(0)
    uni = UnifiedTokenDecoder(cfg).to(device=DEVICE, dtype=DTYPE)

    return cfg, ref, uni


def demo_env_info() -> None:
    _require_cuda_bf16()
    print("\n==================== demo_env_info ====================")
    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"device_name={torch.cuda.get_device_name(0)}")
    print(f"bf16_supported={torch.cuda.is_bf16_supported()}")
    print(f"DEVICE={DEVICE}, DTYPE={DTYPE}")


def demo_unified_token_decoder_forward_bf16_cuda() -> None:
    print("\n==================== demo_unified_token_decoder_forward_bf16_cuda ====================")
    cfg, _ref, uni = _make_models()
    uni.eval()

    x = torch.randn(2, 16, cfg.d_model, device=DEVICE, dtype=DTYPE)
    print(f"[input]  {_tensor_stat(x)}")

    with torch.inference_mode():
        y = uni(x)

    print(f"[output] {_tensor_stat(y)}")

    assert y.shape == (2, 16, cfg.d_model)
    assert torch.isfinite(y.float()).all()
    print("[ok] forward pass (cuda+bf16)")


def demo_convert_and_revert_state_dict(*, reduce_mode: bool = False) -> None:
    print("\n==================== demo_convert_and_revert_state_dict ====================")
    print(f"reduce_mode={reduce_mode}")

    _cfg, ref, _uni = _make_models(reduce=reduce_mode)

    # convert/revert 是 key/shape/拼接映射：这里用 GPU+bf16 的 state_dict 跑一遍
    ref_sd = _to_cuda_bf16(copy.deepcopy(ref.state_dict()))
    print(f"[ref] state_dict keys={len(ref_sd)}")

    # 原始 -> Unified
    uni_sd = UnifiedTokenDecoder.convert_hf_state_dict(ref_sd, reduce_mode=reduce_mode)
    print(f"[uni] converted state_dict keys={len(uni_sd)}")

    # Unified -> 原始
    ref_sd_2 = UnifiedTokenDecoder.revert_hf_state_dict(uni_sd, reduce_mode=reduce_mode)
    print(f"[ref2] reverted state_dict keys={len(ref_sd_2)}")

    # key 必须完全一致
    missing = sorted(set(ref_sd.keys()) - set(ref_sd_2.keys()))
    extra = sorted(set(ref_sd_2.keys()) - set(ref_sd.keys()))
    print(f"[check] missing_keys={len(missing)}, extra_keys={len(extra)}")
    if missing:
        print("  missing:")
        for k in missing[:20]:
            print("   -", k)
    if extra:
        print("  extra:")
        for k in extra[:20]:
            print("   -", k)
    assert not missing and not extra

    # tensor 必须完全一致（转换/逆转换应该是可逆的）
    max_diff = 0.0
    max_diff_k: str | None = None
    for k in ref_sd.keys():
        a = ref_sd[k]
        b = ref_sd_2[k]
        if not torch.equal(a, b):
            diff = (a.float() - b.float()).abs().max().item()
            if diff > max_diff:
                max_diff = diff
                max_diff_k = k

    print(f"[check] max_diff={max_diff}, max_diff_key={max_diff_k}")
    assert max_diff == 0.0
    print("[ok] convert_hf_state_dict <-> revert_hf_state_dict are reversible")


def demo_load_converted_weights_and_forward(*, reduce_mode: bool = False) -> None:
    """把 ref 权重 convert 成 unified 格式 load 进 UnifiedTokenDecoder，然后跑 forward（cuda+bf16）。"""

    print("\n==================== demo_load_converted_weights_and_forward ====================")
    print(f"reduce_mode={reduce_mode}")

    cfg, ref, uni = _make_models(reduce=reduce_mode)

    ref_sd = _to_cuda_bf16(copy.deepcopy(ref.state_dict()))
    uni_sd = UnifiedTokenDecoder.convert_hf_state_dict(ref_sd, reduce_mode=reduce_mode)

    missing, unexpected = uni.load_state_dict(uni_sd, strict=False)
    print(f"[load] missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print("  missing:")
        for k in missing[:30]:
            print("   -", k)
    if unexpected:
        print("  unexpected:")
        for k in unexpected[:30]:
            print("   -", k)

    uni.eval()
    x = torch.randn(2, 16, cfg.d_model, device=DEVICE, dtype=DTYPE)

    with torch.inference_mode():
        y = uni(x)

    print(f"[forward after load] output: {_tensor_stat(y)}")
    assert torch.isfinite(y.float()).all()
    print("[ok] forward after loading converted weights")


if __name__ == "__main__":
    torch.set_printoptions(precision=4, sci_mode=False)

    demo_env_info()
    demo_unified_token_decoder_forward_bf16_cuda()
    demo_convert_and_revert_state_dict(reduce_mode=False)
    demo_load_converted_weights_and_forward(reduce_mode=False)

    # 如果你也想顺便验证 reduce=True 的分支，可以打开：
    # demo_convert_and_revert_state_dict(reduce_mode=True)
    # demo_load_converted_weights_and_forward(reduce_mode=True)
