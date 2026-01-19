"""UnifiedTokenDecoder demo + 权重转换/回滚验证。

注意：你要求的是 demo 脚本风格（非 pytest）。
- 直接 `python tests/models/keye_ar/test_unified_token_decoder_revert.py` 即可。
- 里面会在 `__main__` 里依次调用各个 demo，并打印调试信息。
"""

from __future__ import annotations

import copy

import torch

from muse.config.model_config import UnifiedTokenDecoderConfig
from muse.models.keye_ar.unified_token_decoder import UnifiedTokenDecoder
from tests.models.keye_ar.original_transformer_decoder import PureDecoderTransformer


def _tensor_stat(x: torch.Tensor) -> str:
    return (
        f"shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device}, "
        f"min={x.min().item():.6g}, max={x.max().item():.6g}, mean={x.float().mean().item():.6g}"
    )


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
        attention_function="sdpa",
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
    )

    torch.manual_seed(0)
    uni = UnifiedTokenDecoder(cfg)
    return cfg, ref, uni


def demo_unified_token_decoder_forward_cpu() -> None:
    print("\n==================== demo_unified_token_decoder_forward_cpu ====================")
    cfg, _ref, uni = _make_models()
    uni.eval()

    x = torch.randn(2, 16, cfg.d_model)
    print(f"[input]  {_tensor_stat(x)}")

    y = uni(x)
    print(f"[output] {_tensor_stat(y)}")

    assert y.shape == (2, 16, cfg.d_model)
    assert torch.isfinite(y).all()
    print("[ok] forward pass")


def demo_convert_and_revert_state_dict(*, reduce_mode: bool = False) -> None:
    print("\n==================== demo_convert_and_revert_state_dict ====================")
    print(f"reduce_mode={reduce_mode}")

    _cfg, ref, _uni = _make_models(reduce=reduce_mode)

    # 参考模型的 state_dict（模拟“原始实现”的 HF 权重命名）
    ref_sd = copy.deepcopy(ref.state_dict())
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
    max_diff_k = None
    for k in ref_sd.keys():
        a = ref_sd[k]
        b = ref_sd_2[k]
        if not torch.equal(a, b):
            diff = (a - b).abs().max().item()
            if diff > max_diff:
                max_diff = diff
                max_diff_k = k

    print(f"[check] max_diff={max_diff}, max_diff_key={max_diff_k}")
    assert max_diff == 0.0
    print("[ok] convert_hf_state_dict <-> revert_hf_state_dict are reversible")


def demo_load_converted_weights_and_forward(*, reduce_mode: bool = False) -> None:
    """额外 demo：把 ref 权重转成 unified 格式 load 进 UnifiedTokenDecoder，然后跑一次 forward。

    这个 demo 的价值：确认 convert 出来的 key/shape 能被 UnifiedTokenDecoder.load_state_dict 接受。
    """

    print("\n==================== demo_load_converted_weights_and_forward ====================")
    print(f"reduce_mode={reduce_mode}")

    cfg, ref, uni = _make_models(reduce=reduce_mode)

    ref_sd = copy.deepcopy(ref.state_dict())
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
    x = torch.randn(2, 16, cfg.d_model)
    y = uni(x)
    print(f"[forward after load] output: {_tensor_stat(y)}")
    assert torch.isfinite(y).all()
    print("[ok] forward after loading converted weights")


if __name__ == "__main__":
    torch.set_printoptions(precision=4, sci_mode=False)

    demo_unified_token_decoder_forward_cpu()
    demo_convert_and_revert_state_dict(reduce_mode=False)
    demo_load_converted_weights_and_forward(reduce_mode=False)

    # 如果你也想顺便验证 reduce=True 的分支，可以打开：
    # demo_convert_and_revert_state_dict(reduce_mode=True)
    # demo_load_converted_weights_and_forward(reduce_mode=True)
