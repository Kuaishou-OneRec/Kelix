"""
Keye-VL Weight Loading Debugger
===============================
Directly compares model parameters against the source checkpoint file to verify loading correctness.
"""

import os
import sys
import logging
import glob
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn

# === 导入 Muse 模型 ===
from muse.models.keye_tokenizer_video import modeling as muse_mod
from muse.models.keye_tokenizer_video import modeling_keye_origin as origin_mod
from muse.config import Qwen3Config, KeyeVisionConfig, KeyeTokenizerConfig
from muse.training.common import set_default_dtype

logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

DEFAULT_CKPT = "/mmu_mllm_hdd_2/maosiyang/output/Keye/vq_end2end_video/discrete/run_exp0.0.1_stage1_baseline/step16000/global_step16000/converted"

def _load_checkpoint_robust(path_str: str, device="cpu") -> Dict[str, torch.Tensor]:
    path = Path(path_str)
    if path.is_file(): return torch.load(path, map_location=device)
    
    state_dict = {}
    from safetensors.torch import safe_open
    
    # Safetensors
    st_files = sorted(glob.glob(str(path / "*.safetensors")))
    if st_files:
        logger.info(f"Loading {len(st_files)} safetensors files...")
        for f in st_files:
            with safe_open(f, framework="pt", device=device) as open_f:
                for k in open_f.keys(): state_dict[k] = open_f.get_tensor(k)
        return state_dict
    
    # Bin
    bin_files = sorted(glob.glob(str(path / "*.bin")))
    if bin_files:
        logger.info(f"Loading {len(bin_files)} bin files...")
        for f in bin_files:
            if "training_args" in f: continue
            part = torch.load(f, map_location=device)
            if "module" in part: part = part["module"]
            state_dict.update(part)
        return state_dict

    raise ValueError(f"No weights found in {path_str}")

def _load_config_json(ckpt_path: str) -> Dict[str, Any]:
    p = Path(ckpt_path)
    base_dir = p if p.is_dir() else p.parent
    with open(base_dir / "config.json", "r") as f: return json.load(f)

import json

def check_weight_consistency(model_name: str, model: nn.Module, ckpt_state_dict: Dict[str, torch.Tensor]):
    logger.info(f"\n{'='*80}")
    logger.info(f"Checking Weights for: {model_name}")
    logger.info(f"{'='*80}")
    
    model_state = model.state_dict()
    
    # Statistics
    total_params = len(model_state)
    matched_exact_name = 0
    matched_value = 0
    mismatched_value = 0
    not_found_in_ckpt = 0
    
    # Sample errors
    mismatches = []
    missing = []
    
    # 1. 遍历模型参数
    for name, param in model_state.items():
        # 跳过 buffer 中不需要梯度的 (如 position_ids)
        if not param.is_floating_point(): continue 
        
        # 尝试在 Checkpoint 中找到对应的 Key
        # Muse 需要反向查找 Convert 逻辑，这里比较难，所以我们主要检查 Origin
        # 或者尝试模糊匹配
        
        target_k = None
        
        # === 简单匹配逻辑 ===
        # 1. 直接匹配
        if name in ckpt_state_dict:
            target_k = name
        # 2. Origin 常见前缀差异 (siglip.vision_model vs visual_tokenizer...)
        else:
            # 这是一个启发式搜索，尝试找到 ckpt 中对应的 key
            # 你可以根据实际情况添加更多规则
            pass

        # 另外一种方式：如果提供了 convert_hf_state_dict，我们可以反向推导
        # 但这里为了简单，我们只对比那些名字能对上的，或者在 Convert 过程中我们手动映射过的
        
        # 对于 Muse，我们假设 Convert 逻辑是： ckpt_key -> muse_key
        # 所以我们应该遍历 ckpt_keys，看转换后的 key 是否在 muse model 中，且数值一致
        pass 

    # === 反向检查策略 ===
    # 遍历 Checkpoint 中的 Key，经过 Convert 后，去 Model 里查值
    
    # 如果是 Origin 模型，通常 Key 变化不大
    # 如果是 Muse 模型，使用了 convert_hf_state_dict
    
    logger.info(f"Model Parameters: {len(model_state)}")
    logger.info(f"Checkpoint Keys : {len(ckpt_state_dict)}")
    
    if hasattr(model, "convert_hf_state_dict"):
        logger.info("Using model.convert_hf_state_dict for mapping...")
        # 模拟转换过程
        mapped_ckpt = model.convert_hf_state_dict(ckpt_state_dict, tie_word_embeddings=True)
        
        for muse_key, muse_val in model_state.items():
            if muse_key in mapped_ckpt:
                ckpt_val = mapped_ckpt[muse_key].to(muse_val.device)
                
                # Shape check
                if muse_val.shape != ckpt_val.shape:
                     # 尝试转置 (Linear weights)
                     if muse_val.shape == ckpt_val.t().shape:
                         ckpt_val = ckpt_val.t()
                
                if muse_val.shape != ckpt_val.shape:
                    logger.error(f"❌ Shape Mismatch: {muse_key} | Model {muse_val.shape} != Ckpt {ckpt_val.shape}")
                    continue
                    
                diff = (muse_val - ckpt_val).abs().max().item()
                if diff > 1e-3:
                    mismatches.append((muse_key, diff, muse_val.mean().item(), ckpt_val.mean().item()))
                    mismatched_value += 1
                else:
                    matched_value += 1
            else:
                # 忽略一些统计量 buffer
                if "running_mean" in muse_key or "running_var" in muse_key or "num_batches_tracked" in muse_key:
                    continue
                not_found_in_ckpt += 1
                missing.append(muse_key)
                
    else:
        # Origin Model (通常不需要复杂转换，或者转换逻辑在内部)
        # 我们尝试直接匹配
        for orig_key, orig_val in model_state.items():
            # Origin 模型加载时可能加了前缀，或者 ckpt 里有前缀
            # Ckpt: visual_tokenizer.visual.vision_model.embeddings...
            # Model: visual_tokenizer.visual.embeddings... (如果是 SiglipVisionModel)
            
            # 暴力尝试匹配
            candidates = [
                orig_key, 
                f"siglip.{orig_key}", 
                f"visual_tokenizer.{orig_key}",
                f"model.{orig_key}"
            ]
            
            # 也要处理 ckpt key 比 model key 长的情况
            # 例如 model key: visual.embeddings...
            # ckpt key: visual_tokenizer.visual.vision_model.embeddings...
            
            found = False
            for k in ckpt_state_dict:
                if k.endswith(orig_key): # 简单的后缀匹配
                    # 进一步确认
                    # 如果 orig_key 很短 (e.g. "weight")，后缀匹配不可靠
                    if len(orig_key) < 10: continue
                    
                    ckpt_val = ckpt_state_dict[k].to(orig_val.device)
                    if ckpt_val.shape == orig_val.shape:
                        diff = (orig_val - ckpt_val).abs().max().item()
                        if diff > 1e-3:
                            mismatches.append((orig_key, diff, orig_val.mean().item(), ckpt_val.mean().item()))
                            mismatched_value += 1
                        else:
                            matched_value += 1
                        found = True
                        break
            
            if not found:
                not_found_in_ckpt += 1
                missing.append(orig_key)

    # Report
    logger.info(f"✅ Matched Weights: {matched_value}")
    logger.info(f"❌ Mismatched Weights: {mismatched_value}")
    logger.info(f"❓ Not found in Checkpoint: {not_found_in_ckpt}")
    
    if mismatches:
        logger.info("\nTop 10 Mismatches (Key | Diff | Model Mean | Ckpt Mean):")
        for m in mismatches[:10]:
            logger.info(f"  {m[0]:<60} | {m[1]:.2e} | {m[2]:.4f} | {m[3]:.4f}")
            
    if missing:
        logger.info("\nTop 10 Missing Keys (Present in Model, Absent in Ckpt):")
        for m in missing[:10]:
            logger.info(f"  {m}")

def main():
    ckpt_path = DEFAULT_CKPT
    device = "cpu" # 权重对比用 CPU 足够，且显存更省
    dtype = torch.float16
    
    logger.info(f"Loading Checkpoint from {ckpt_path} ...")
    ckpt_state = _load_checkpoint_robust(ckpt_path, device=device)
    
    # 统一转 dtype
    for k, v in ckpt_state.items():
        ckpt_state[k] = v.to(dtype)

    raw_cfg = _load_config_json(ckpt_path)
    
    # --- 构建 Muse Config ---
    qwen_cfg = Qwen3Config(
        model_class="Qwen3Model",
        vocab_size=raw_cfg["vocab_size"],
        embed_dim=raw_cfg["hidden_size"],
        num_layers=raw_cfg["num_hidden_layers"],
        num_heads=raw_cfg["num_attention_heads"],
        num_kv_heads=raw_cfg["num_key_value_heads"],
        head_dim=raw_cfg["head_dim"],
        intermediate_dim=raw_cfg["intermediate_size"],
        max_seq_len=raw_cfg["max_position_embeddings"],
        rope_base=float(raw_cfg.get("rope_theta", 1_000_000)),
        attention_function="flash_attention_2",
        tie_word_embeddings=raw_cfg.get("tie_word_embeddings", True),
    )
    
    outer_vcfg = raw_cfg["vision_config"]
    inner_vcfg = outer_vcfg["vision_config"]
    vision_cfg = KeyeVisionConfig(
        hidden_size=inner_vcfg["hidden_size"],
        num_hidden_layers=inner_vcfg["num_hidden_layers"],
        num_attention_heads=inner_vcfg["num_attention_heads"],
        image_size=inner_vcfg["image_size"],
        patch_size=inner_vcfg["patch_size"],
        intermediate_size=inner_vcfg["intermediate_size"],
        has_learnable_position_embedding=inner_vcfg.get("has_learnable_position_embedding", True),
        attention_function="flash_attention_2",
    )
    tokenizer_cfg = KeyeTokenizerConfig(
        vision_config=vision_cfg,
        llm_hidden_size=outer_vcfg.get("llm_hidden_size", 4096),
        embedding_dim=outer_vcfg.get("embedding_dim", 128),
        init_embedding_dim=outer_vcfg.get("init_embedding_dim", 4096),
        codebook_size=outer_vcfg.get("codebook_size", 65536),
        n_q_tokens=outer_vcfg.get("n_q_tokens", 8),
        split_voc=outer_vcfg.get("split_voc", 1),
        add_voc_reducer=outer_vcfg.get("add_voc_reducer", False),
        split_dim=outer_vcfg.get("split_dim", False),
        vq_sampling_mode="argmin",
    )

    # 1. Initialize Muse
    logger.info("Initializing Muse Model...")
    with set_default_dtype(dtype):
        muse_model = muse_mod.KeyeForConditionalGeneration(
            qwen_config=qwen_cfg,
            vision_config=vision_cfg,
            tokenizer_config=tokenizer_cfg,
            image_token_id=raw_cfg.get("image_token_id", 151655),
            pool="sum"
        )
    
    logger.info("Loading Muse Weights...")
    # 使用转换逻辑加载
    muse_converted = muse_model.convert_hf_state_dict(ckpt_state, tie_word_embeddings=qwen_cfg.tie_word_embeddings)
    muse_model.load_state_dict(muse_converted, strict=False)
    
    # Check Muse
    check_weight_consistency("Muse Model", muse_model, ckpt_state)

    # 2. Initialize Origin (如果不关心 Origin 可以注释掉，但为了对比最好保留)
    logger.info("\nInitializing Origin Model...")
    origin_cfg = origin_mod.KeyeConfig.from_pretrained(ckpt_path)
    
    with set_default_dtype(dtype):
        origin_model = origin_mod.KeyeForConditionalGeneration(origin_cfg)
        
    logger.info("Loading Origin Weights...")
    origin_model.load_state_dict(ckpt_state, strict=False)
    
    # Check Origin
    # Origin 的结构比较深，check_weight_consistency 的简单后缀匹配可能不够
    # 但我们主要看 visual 部分
    check_weight_consistency("Origin Model", origin_model, ckpt_state)

if __name__ == "__main__":
    main()