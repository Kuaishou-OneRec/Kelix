# Cross Attention Bug修复

## 修复日期
2025-10-21

## 问题描述

测试显示5个 `TransformerCrossAttentionLayer` 测试失败：
```
RuntimeError: The size of tensor a (8) must match the size of tensor b (12) at non-singleton dimension 3
```

## 根本原因

问题出在 `EagerAttention.forward` 方法中对 `mask` 参数的处理。

在 cross attention 中：
- Query (q) 来自 decoder: `[b, n_h, s_x, h_d]` = `[2, 4, 8, 16]`
- Key (k) 来自 encoder: `[b, n_h, s_y, h_d]` = `[2, 4, 12, 16]`
- Value (v) 来自 encoder: `[b, n_h, s_y, h_d]` = `[2, 4, 12, 16]`
- Encoder mask 传入时形状: `[b, s_x, s_y]` = `[2, 8, 12]`
- Scores 形状: `[b, n_h, s_x, s_y]` = `[2, 4, 8, 12]`

当尝试将mask应用到scores时（`scores + mask`），由于mask缺少head维度，导致广播失败并抛出维度不匹配错误。

## 修复方案

修改 `muse/layers/attention_utils.py` 中的 `EagerAttention.forward` 方法，在应用mask之前检查其维度：

### 修复前
```python
# Apply custom mask (if provided)
if mask is not None:
    scores = scores + mask
```

### 修复后
```python
# Apply custom mask (if provided)
if mask is not None:
    # mask shape: [b, s_q, s_k] or [b, n_h, s_q, s_k]
    # scores shape: [b, n_h, s_q, s_k]
    # If mask doesn't have the head dimension, unsqueeze it
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)  # [b, 1, s_q, s_k]
    scores = scores + mask
```

## 修复逻辑

1. **检查mask维度**: 检查mask是否有4个维度（包含head维度）
2. **扩展维度**: 如果mask只有3个维度，在维度1处unsqueeze，添加head维度
3. **广播应用**: PyTorch会自动将 `[b, 1, s_q, s_k]` 广播到 `[b, n_h, s_q, s_k]`

## 影响的测试

修复后，以下5个测试应该通过：
- ✅ `test_forward_basic`
- ✅ `test_forward_different_encoder_seq_len`
- ✅ `test_gradient_flow`
- ✅ `test_residual_connections`
- ✅ `test_with_scale_modules`

## 验证命令

```bash
cd /Users/zhouyang12/code/muse

# 安装依赖
pip install torch pytest

# 运行所有测试
python -m pytest tests/ -v

# 运行特定测试验证修复
python -m pytest tests/test_transformer.py::TestTransformerCrossAttentionLayer -v
```

## 总结

这个修复解决了 cross attention 中mask维度不匹配的问题，使得 `TransformerCrossAttentionLayer` 能够正确处理encoder mask。

修复后，所有132个测试应该能够通过（除了被跳过的FlashAttention2测试）。

