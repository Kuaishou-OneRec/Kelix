#!/usr/bin/env python3
"""
对比两个仓库模型的前向 logits 和后向梯度

此脚本使用相同的输入数据，分别在两个仓库的模型上运行前向和后向传播，
然后对比输出的 logits 和梯度是否一致。

使用方法:
    # 在 end2end 目录下运行:
    cd /llm_reco/maosiyang/video_tok/muse
    python verify_forward_backward.py --mode end2end
    
    # 在 msy_master_2 目录下运行:
    cd /llm_reco/maosiyang/msy_master_2/muse
    python verify_forward_backward.py --mode msy_master_2
    
    # 对比结果:
    python verify_forward_backward.py --mode compare
"""

import argparse
import json
import os
import sys
import numpy as np
from typing import Optional, Tuple, Dict, Any
import torch

# 固定的测试样本
TEST_VIDEO_PATH = "/llm_reco/maosiyang/23b77760a4304e9092eb3b45b7bf8050.mp4"
TEST_MESSAGES = [
    {
        "role": "user",
        "content": [
            {"type": "video", "video": TEST_VIDEO_PATH},
        ],
    },
    {
        "role": "assistant",
        "content": "i am an apple"
    }
]

# 使用相同的模型路径
MODEL_PATH_END2END = "/llm_reco_ssd/maosiyang/models/muse/keye_tokenizer_video_baseline"
MODEL_PATH_MSY_MASTER_2 = "/llm_reco_ssd/maosiyang/models/muse/keye_tokenizer_end2end_image_for_stage_2_video"

# 输出目录
OUTPUT_DIR = "/llm_reco/maosiyang/temp"


def save_tensor_summary(tensor, name, summary_dict):
    """保存 tensor 的统计信息"""
    if tensor is None:
        summary_dict[f"{name}"] = None
        return
    
    arr = tensor.detach().cpu().float().numpy()
    summary_dict[f"{name}_shape"] = list(arr.shape)
    summary_dict[f"{name}_dtype"] = str(tensor.dtype)
    summary_dict[f"{name}_sum"] = float(arr.sum())
    summary_dict[f"{name}_mean"] = float(arr.mean()) if arr.size > 0 else 0
    summary_dict[f"{name}_min"] = float(arr.min()) if arr.size > 0 else 0
    summary_dict[f"{name}_max"] = float(arr.max()) if arr.size > 0 else 0
    summary_dict[f"{name}_std"] = float(arr.std()) if arr.size > 0 else 0
    summary_dict[f"{name}_numel"] = int(arr.size)
    
    # 保存前1000个元素用于精确对比
    flat = arr.flatten()[:1000]
    summary_dict[f"{name}_first1000"] = flat.tolist()
    
    # 保存全部数据（限制大小）
    flat_all = arr.flatten()
    if len(flat_all) <= 100000:
        summary_dict[f"{name}_all"] = flat_all.tolist()
    else:
        summary_dict[f"{name}_all_truncated"] = True
        summary_dict[f"{name}_all"] = flat_all[:100000].tolist()


def save_gradients(model, summary_dict, prefix="grad"):
    """保存模型中所有参数的梯度"""
    grad_info = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad.detach().cpu().float()
            grad_info[f"{prefix}/{name}_shape"] = list(grad.shape)
            grad_info[f"{prefix}/{name}_sum"] = float(grad.sum().item())
            grad_info[f"{prefix}/{name}_mean"] = float(grad.mean().item())
            grad_info[f"{prefix}/{name}_max"] = float(grad.max().item())
            grad_info[f"{prefix}/{name}_min"] = float(grad.min().item())
            grad_info[f"{prefix}/{name}_std"] = float(grad.std().item())
            grad_info[f"{prefix}/{name}_norm"] = float(grad.norm().item())
            # 保存前100个元素
            grad_info[f"{prefix}/{name}_first100"] = grad.flatten()[:100].tolist()
    
    summary_dict["gradients"] = grad_info
    return grad_info


def prepare_batch_end2end():
    """在 end2end 环境下准备输入 batch"""
    from transformers import AutoProcessor
    from recovlm.models.tokenizer_end2end_mt_1drope_video.keye_vl_utils import process_vision_info
    
    processor = AutoProcessor.from_pretrained(MODEL_PATH_END2END, trust_remote_code=True)
    
    text = processor.apply_chat_template(
        TEST_MESSAGES,
        tokenize=False,
        add_generation_prompt=False
    )
    text += "<|endoftext|>"
    
    image_inputs, video_inputs = process_vision_info(TEST_MESSAGES)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=False,
        truncation=False,
        return_tensors="pt",
    )
    
    # 计算 loss_mask
    input_ids = inputs['input_ids'].flatten()
    loss_mask = torch.ones_like(input_ids, dtype=torch.float32)
    
    vision_start_id = 151652
    vision_end_id = 151653
    
    in_vision = False
    for i in range(len(input_ids)):
        if input_ids[i] == vision_start_id:
            in_vision = True
        if in_vision:
            loss_mask[i] = 0
        if input_ids[i] == vision_end_id:
            in_vision = False
    
    inputs['loss_mask'] = loss_mask.unsqueeze(0)
    inputs['cu_seqlens'] = torch.tensor([0, inputs['input_ids'].shape[1]], dtype=torch.int32)
    
    return inputs, text


def prepare_batch_msy_master_2():
    """在 msy_master_2 环境下准备输入 batch"""
    from transformers import AutoProcessor
    
    try:
        from tests.models.tokenizer_end2end_mt_1drope_video.keye_vl_utils import process_vision_info
    except ImportError:
        from muse.models.keye_tokenizer_end2end_video.keye_vl_utils import process_vision_info
    
    processor = AutoProcessor.from_pretrained(MODEL_PATH_MSY_MASTER_2, trust_remote_code=True)
    
    text = processor.apply_chat_template(
        TEST_MESSAGES,
        tokenize=False,
        add_generation_prompt=False
    )
    text += "<|endoftext|>"
    
    image_inputs, video_inputs = process_vision_info(TEST_MESSAGES)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=False,
        truncation=False,
        return_tensors="pt",
    )
    
    # 计算 loss_mask
    input_ids = inputs['input_ids'].flatten()
    loss_mask = torch.ones_like(input_ids, dtype=torch.float32)
    
    vision_start_id = 151652
    vision_end_id = 151653
    
    in_vision = False
    for i in range(len(input_ids)):
        if input_ids[i] == vision_start_id:
            in_vision = True
        if in_vision:
            loss_mask[i] = 0
        if input_ids[i] == vision_end_id:
            in_vision = False
    
    inputs['loss_mask'] = loss_mask.unsqueeze(0)
    inputs['cu_seqlens'] = torch.tensor([0, inputs['input_ids'].shape[1]], dtype=torch.int32)
    
    return inputs, text


def run_end2end():
    """在 end2end 环境下运行前向和后向"""
    print("=" * 70)
    print("Running in end2end mode - Forward and Backward")
    print("=" * 70)
    
    sys.path.insert(0, os.getcwd())
    
    os.environ['nosp'] = '1'
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['LOCAL_RANK'] = '0'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    
    import torch
    import torch.distributed as dist
    
    if not dist.is_initialized():
        dist.init_process_group(backend='gloo', init_method='env://')
    
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    try:
        print(f"\n📁 Model path: {MODEL_PATH_END2END}")
        
        # 准备输入数据
        print("\n🔄 Preparing input batch...")
        inputs, text = prepare_batch_end2end()
        
        print(f"   Input keys: {list(inputs.keys())}")
        for key, value in inputs.items():
            if value is not None and hasattr(value, 'shape'):
                print(f"   {key}: shape={value.shape}, dtype={value.dtype}")
        
        # 加载模型
        print("\n⚙️ Loading Model...")
        from recovlm.models.tokenizer_end2end_mt_1drope_video.modeling_keye import (
            KeyeForConditionalGeneration as KeyeImageTokenizer_end2end_mt_1drope_video
        )
        from recovlm.training.common import set_default_dtype
        
        with set_default_dtype(torch.bfloat16):
            model = KeyeImageTokenizer_end2end_mt_1drope_video.from_pretrained(
                MODEL_PATH_END2END,
                _attn_implementation="flash_attention_2",
                use_cache=False,
                ignore_mismatched_sizes=True,
            )
        
        model = model.cuda().bfloat16()
        model.train()
        print(f"   Model loaded: {type(model).__name__}")
        
        # 将输入移到 GPU
        print("\n🔄 Moving inputs to GPU...")
        batch = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if v.is_floating_point():
                    batch[k] = v.to(device='cuda', dtype=torch.bfloat16)
                else:
                    batch[k] = v.to(device='cuda')
            else:
                batch[k] = v
        
        # 准备 labels
        input_ids = batch["input_ids"]
        loss_mask = batch.get("loss_mask", None)
        
        input_ids = input_ids * (input_ids > 0).to(torch.int64)
        if loss_mask is not None:
            labels = input_ids * loss_mask.to(torch.int64) + (-100) * (1 - loss_mask.to(torch.int64))
        else:
            labels = input_ids.clone()
        
        # 前向传播
        print("\n🚀 Running Forward Pass...")
        output = model(
            input_ids=input_ids,
            attention_mask=batch.get("attention_mask", None),
            position_ids=batch.get("position_ids", None),
            pixel_values=batch.get("pixel_values", None),
            image_grid_thw=batch.get("image_grid_thw", None),
            pixel_values_videos=batch.get("pixel_values_videos", None),
            video_grid_thw=batch.get("video_grid_thw", None),
            cu_seqlens=batch.get("cu_seqlens", None),
        )
        
        logits = output.logits
        print(f"   Logits shape: {logits.shape}")
        
        # 计算 loss
        print("\n🔄 Computing Loss...")
        from recovlm.losses import CrossEntropyLoss
        loss_fn = CrossEntropyLoss(ignore_index=-100, return_token_loss=True, shift_labels=False)
        
        # shift labels
        pad = torch.full((labels.shape[0], 1), -100, dtype=labels.dtype).to(device=labels.device)
        shifted_labels = torch.cat([labels[:, 1:], pad], dim=-1)
        
        lm_loss, per_token_loss = loss_fn(logits=logits, labels=shifted_labels)
        
        # 添加 codebook loss
        codebook_loss = output.codebook_loss
        commitment_loss = output.commitment_loss
        
        total_loss = lm_loss + 1.0 * (sum(codebook_loss)/len(codebook_loss) + 
                                       0.25 * sum(commitment_loss)/len(commitment_loss))
        
        print(f"   LM Loss: {lm_loss.item():.6f}")
        print(f"   Total Loss: {total_loss.item():.6f}")
        
        # 后向传播
        print("\n🔙 Running Backward Pass...")
        model.zero_grad()
        total_loss.backward()
        
        # 收集结果
        summary = {
            "model_path": MODEL_PATH_END2END,
            "text_prompt": text,
            "lm_loss": lm_loss.item(),
            "total_loss": total_loss.item(),
            "codebook_loss": [x.item() for x in codebook_loss],
            "commitment_loss": [x.item() for x in commitment_loss],
        }
        
        # 保存 logits 信息
        save_tensor_summary(logits, "logits", summary)
        
        # 保存 per_token_loss
        save_tensor_summary(per_token_loss, "per_token_loss", summary)
        
        # 保存梯度
        save_gradients(model, summary)
        
        # 保存输入信息
        save_tensor_summary(batch["input_ids"], "input_ids", summary)
        if batch.get("pixel_values_videos") is not None:
            save_tensor_summary(batch["pixel_values_videos"], "pixel_values_videos", summary)
        
        # 保存到文件
        output_file = os.path.join(OUTPUT_DIR, "forward_backward_end2end.json")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Results saved to {output_file}")
        
        # 打印关键信息
        print("\n📊 Key Results:")
        print(f"   Logits sum: {logits.sum().item():.6f}")
        print(f"   Logits mean: {logits.mean().item():.6f}")
        print(f"   Logits std: {logits.std().item():.6f}")
        
        # 统计梯度
        total_grad_norm = 0.0
        grad_count = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.norm().item() ** 2
                grad_count += 1
        total_grad_norm = total_grad_norm ** 0.5
        print(f"   Total gradient norm: {total_grad_norm:.6f}")
        print(f"   Number of params with gradients: {grad_count}")
        
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()


def run_msy_master_2():
    """在 msy_master_2 环境下运行前向和后向"""
    print("=" * 70)
    print("Running in msy_master_2 mode - Forward and Backward")
    print("=" * 70)
    
    sys.path.insert(0, os.getcwd())
    
    os.environ['nosp'] = '1'
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['LOCAL_RANK'] = '0'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29501'
    
    import torch
    import torch.distributed as dist
    
    if not dist.is_initialized():
        dist.init_process_group(backend='gloo', init_method='env://')
    
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    try:
        print(f"\n📁 Model path: {MODEL_PATH_MSY_MASTER_2}")
        
        # 准备输入数据
        print("\n🔄 Preparing input batch...")
        inputs, text = prepare_batch_msy_master_2()
        
        print(f"   Input keys: {list(inputs.keys())}")
        for key, value in inputs.items():
            if value is not None and hasattr(value, 'shape'):
                print(f"   {key}: shape={value.shape}, dtype={value.dtype}")
        
        # 加载模型
        print("\n⚙️ Loading Model...")
        from muse.models import get_model_class
        from muse.config import load_config
        from muse.training.common import set_default_dtype
        from pathlib import Path
        
        model_config_path = Path(MODEL_PATH_MSY_MASTER_2) / "muse_config.json"
        model_config = load_config(model_config_path)
        model_config.qwen_config.attention_function = "flash_attention_2"
        
        model_cls = get_model_class(model_config.model_class)
        
        with set_default_dtype(torch.bfloat16), torch.device("meta"):
            model = model_cls(model_config)
        
        # 加载权重
        from muse.training.checkpoint import load_hf_checkpoint
        state_dict = load_hf_checkpoint(MODEL_PATH_MSY_MASTER_2)
        
        # 使用简单的 load_state_dict
        model = model.to(torch.bfloat16)
        model.load_state_dict(state_dict, strict=False)
        model = model.cuda()
        model.train()
        
        print(f"   Model loaded: {type(model).__name__}")
        
        # 将输入移到 GPU
        print("\n🔄 Moving inputs to GPU...")
        batch = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if v.is_floating_point():
                    batch[k] = v.to(device='cuda', dtype=torch.bfloat16)
                else:
                    batch[k] = v.to(device='cuda')
            else:
                batch[k] = v
        
        # 准备 labels
        input_ids = batch["input_ids"]
        loss_mask = batch.get("loss_mask", None)
        
        input_ids = input_ids * (input_ids > 0).to(torch.int64)
        if loss_mask is not None:
            labels = input_ids * loss_mask.to(torch.int64) + (-100) * (1 - loss_mask.to(torch.int64))
        else:
            labels = input_ids.clone()
        
        # 前向传播
        print("\n🚀 Running Forward Pass...")
        output = model(
            input_ids=input_ids,
            attention_mask=batch.get("attention_mask", None),
            position_ids=batch.get("position_ids", None),
            pixel_values=batch.get("pixel_values", None),
            image_grid_thw=batch.get("image_grid_thw", None),
            pixel_values_videos=batch.get("pixel_values_videos", None),
            video_grid_thw=batch.get("video_grid_thw", None),
            cu_seqlens=batch.get("cu_seqlens", None),
            labels=labels,
        )
        
        logits = output["logits"]
        print(f"   Logits shape: {logits.shape}")
        
        # 计算 loss
        print("\n🔄 Computing Loss...")
        from muse.losses import CrossEntropyLoss
        loss_fn = CrossEntropyLoss(ignore_index=-100, return_token_loss=True, shift_labels=False)
        
        # shift labels
        pad = torch.full((labels.shape[0], 1), -100, dtype=labels.dtype).to(device=labels.device)
        shifted_labels = torch.cat([labels[:, 1:], pad], dim=-1)
        
        lm_loss, per_token_loss = loss_fn(logits=logits, labels=shifted_labels)
        
        # 添加 codebook loss
        codebook_loss = output.get("codebook_loss", [torch.tensor(0.0)])
        commitment_loss = output.get("commitment_loss", [torch.tensor(0.0)])
        
        if not isinstance(codebook_loss, list):
            codebook_loss = [codebook_loss]
        if not isinstance(commitment_loss, list):
            commitment_loss = [commitment_loss]
        
        total_loss = lm_loss + 1.0 * (sum(codebook_loss)/len(codebook_loss) + 
                                       0.25 * sum(commitment_loss)/len(commitment_loss))
        
        print(f"   LM Loss: {lm_loss.item():.6f}")
        print(f"   Total Loss: {total_loss.item():.6f}")
        
        # 后向传播
        print("\n🔙 Running Backward Pass...")
        model.zero_grad()
        total_loss.backward()
        
        # 收集结果
        summary = {
            "model_path": MODEL_PATH_MSY_MASTER_2,
            "text_prompt": text,
            "lm_loss": lm_loss.item(),
            "total_loss": total_loss.item(),
            "codebook_loss": [x.item() for x in codebook_loss],
            "commitment_loss": [x.item() for x in commitment_loss],
        }
        
        # 保存 logits 信息
        save_tensor_summary(logits, "logits", summary)
        
        # 保存 per_token_loss
        save_tensor_summary(per_token_loss, "per_token_loss", summary)
        
        # 保存梯度
        save_gradients(model, summary)
        
        # 保存输入信息
        save_tensor_summary(batch["input_ids"], "input_ids", summary)
        if batch.get("pixel_values_videos") is not None:
            save_tensor_summary(batch["pixel_values_videos"], "pixel_values_videos", summary)
        
        # 保存到文件
        output_file = os.path.join(OUTPUT_DIR, "forward_backward_msy_master_2.json")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Results saved to {output_file}")
        
        # 打印关键信息
        print("\n📊 Key Results:")
        print(f"   Logits sum: {logits.sum().item():.6f}")
        print(f"   Logits mean: {logits.mean().item():.6f}")
        print(f"   Logits std: {logits.std().item():.6f}")
        
        # 统计梯度
        total_grad_norm = 0.0
        grad_count = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.norm().item() ** 2
                grad_count += 1
        total_grad_norm = total_grad_norm ** 0.5
        print(f"   Total gradient norm: {total_grad_norm:.6f}")
        print(f"   Number of params with gradients: {grad_count}")
        
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()


def compare_results():
    """对比两个模型的结果"""
    print("=" * 70)
    print("Comparing Forward and Backward Results")
    print("=" * 70)
    
    end2end_file = os.path.join(OUTPUT_DIR, "forward_backward_end2end.json")
    msy_file = os.path.join(OUTPUT_DIR, "forward_backward_msy_master_2.json")
    
    if not os.path.exists(end2end_file):
        print(f"Missing: {end2end_file}")
        print("Please run: cd end2end && python verify_forward_backward.py --mode end2end")
        return
    
    if not os.path.exists(msy_file):
        print(f"Missing: {msy_file}")
        print("Please run: cd msy_master_2 && python verify_forward_backward.py --mode msy_master_2")
        return
    
    with open(end2end_file, 'r', encoding='utf-8') as f:
        end2end = json.load(f)
    
    with open(msy_file, 'r', encoding='utf-8') as f:
        msy = json.load(f)
    
    differences = []
    
    # 1. 对比 Loss
    print("\n" + "=" * 50)
    print("## 1. Loss Comparison")
    print("=" * 50)
    
    for loss_name in ['lm_loss', 'total_loss']:
        e2e_val = end2end.get(loss_name, 0)
        msy_val = msy.get(loss_name, 0)
        diff = abs(e2e_val - msy_val)
        rel_diff = diff / max(abs(e2e_val), abs(msy_val), 1e-10)
        
        if rel_diff < 1e-4:
            print(f"  ✅ {loss_name}: end2end={e2e_val:.6f}, msy={msy_val:.6f}")
        else:
            print(f"  ❌ {loss_name}: end2end={e2e_val:.6f}, msy={msy_val:.6f}, diff={diff:.6e} ({rel_diff*100:.2f}%)")
            differences.append(f"{loss_name}")
    
    # 2. 对比 codebook/commitment loss
    print("\n  Codebook losses:")
    e2e_cb = end2end.get('codebook_loss', [])
    msy_cb = msy.get('codebook_loss', [])
    for i, (e, m) in enumerate(zip(e2e_cb, msy_cb)):
        diff = abs(e - m)
        if diff < 1e-4:
            print(f"    ✅ codebook_loss[{i}]: end2end={e:.6f}, msy={m:.6f}")
        else:
            print(f"    ❌ codebook_loss[{i}]: end2end={e:.6f}, msy={m:.6f}, diff={diff:.6e}")
            differences.append(f"codebook_loss_{i}")
    
    print("\n  Commitment losses:")
    e2e_cm = end2end.get('commitment_loss', [])
    msy_cm = msy.get('commitment_loss', [])
    for i, (e, m) in enumerate(zip(e2e_cm, msy_cm)):
        diff = abs(e - m)
        if diff < 1e-4:
            print(f"    ✅ commitment_loss[{i}]: end2end={e:.6f}, msy={m:.6f}")
        else:
            print(f"    ❌ commitment_loss[{i}]: end2end={e:.6f}, msy={m:.6f}, diff={diff:.6e}")
            differences.append(f"commitment_loss_{i}")
    
    # 3. 对比 Logits
    print("\n" + "=" * 50)
    print("## 2. Logits Comparison")
    print("=" * 50)
    
    for stat in ['sum', 'mean', 'min', 'max', 'std']:
        key = f"logits_{stat}"
        e2e_val = end2end.get(key, 0)
        msy_val = msy.get(key, 0)
        diff = abs(e2e_val - msy_val)
        rel_diff = diff / max(abs(e2e_val), abs(msy_val), 1e-10)
        
        if rel_diff < 1e-3:
            print(f"  ✅ logits {stat}: end2end={e2e_val:.6f}, msy={msy_val:.6f}")
        else:
            print(f"  ❌ logits {stat}: end2end={e2e_val:.6f}, msy={msy_val:.6f}, diff={diff:.6e} ({rel_diff*100:.2f}%)")
            differences.append(f"logits_{stat}")
    
    # 详细对比 logits first1000
    e2e_logits = np.array(end2end.get('logits_first1000', []))
    msy_logits = np.array(msy.get('logits_first1000', []))
    
    if len(e2e_logits) > 0 and len(msy_logits) > 0:
        min_len = min(len(e2e_logits), len(msy_logits))
        e2e_logits = e2e_logits[:min_len]
        msy_logits = msy_logits[:min_len]
        
        logits_diff = np.abs(e2e_logits - msy_logits)
        print(f"\n  Logits element-wise comparison (first {min_len}):")
        print(f"    Max abs diff: {logits_diff.max():.6e}")
        print(f"    Mean abs diff: {logits_diff.mean():.6e}")
        
        close_count = np.sum(np.isclose(e2e_logits, msy_logits, rtol=1e-3, atol=1e-5))
        print(f"    Close matches: {close_count}/{min_len} ({close_count/min_len*100:.1f}%)")
    
    # 4. 对比梯度
    print("\n" + "=" * 50)
    print("## 3. Gradient Comparison")
    print("=" * 50)
    
    e2e_grads = end2end.get('gradients', {})
    msy_grads = msy.get('gradients', {})
    
    # 找出共同的参数
    e2e_params = set([k.replace('grad/', '').rsplit('_', 1)[0] for k in e2e_grads.keys() if k.endswith('_norm')])
    msy_params = set([k.replace('grad/', '').rsplit('_', 1)[0] for k in msy_grads.keys() if k.endswith('_norm')])
    
    common_params = e2e_params & msy_params
    only_e2e = e2e_params - msy_params
    only_msy = msy_params - e2e_params
    
    print(f"\n  Common parameters with gradients: {len(common_params)}")
    if only_e2e:
        print(f"  ⚠️ Only in end2end: {len(only_e2e)}")
    if only_msy:
        print(f"  ⚠️ Only in msy_master_2: {len(only_msy)}")
    
    # 对比梯度 norm
    print("\n  Gradient norm comparison (sample):")
    grad_diffs = []
    for param in sorted(list(common_params))[:20]:  # 只显示前20个
        e2e_norm = e2e_grads.get(f"grad/{param}_norm", 0)
        msy_norm = msy_grads.get(f"grad/{param}_norm", 0)
        diff = abs(e2e_norm - msy_norm)
        rel_diff = diff / max(e2e_norm, msy_norm, 1e-10)
        grad_diffs.append((param, e2e_norm, msy_norm, diff, rel_diff))
    
    # 按相对差异排序
    grad_diffs.sort(key=lambda x: -x[4])
    
    for param, e2e_norm, msy_norm, diff, rel_diff in grad_diffs[:10]:
        if rel_diff < 0.01:
            status = "✅"
        elif rel_diff < 0.1:
            status = "⚠️"
        else:
            status = "❌"
            differences.append(f"grad_{param}")
        
        print(f"    {status} {param[-50:]}: e2e={e2e_norm:.4e}, msy={msy_norm:.4e}, rel_diff={rel_diff*100:.1f}%")
    
    # 5. 总结
    print("\n" + "=" * 70)
    print("## SUMMARY")
    print("=" * 70)
    
    if len(differences) == 0:
        print("🎉 ALL CHECKS PASSED! Forward and backward results are consistent.")
    else:
        print(f"⚠️ Found {len(differences)} significant differences:")
        for diff in differences[:20]:
            print(f"   - {diff}")
        if len(differences) > 20:
            print(f"   ... and {len(differences) - 20} more")
        print("\nThese differences may explain the training behavior discrepancy.")


def main():
    parser = argparse.ArgumentParser(description='Verify forward/backward equivalence between two repos')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['end2end', 'msy_master_2', 'compare'],
                        help='Run mode')
    args = parser.parse_args()
    
    if args.mode == 'end2end':
        run_end2end()
    elif args.mode == 'msy_master_2':
        run_msy_master_2()
    elif args.mode == 'compare':
        compare_results()


if __name__ == "__main__":
    main()

