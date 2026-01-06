#!/usr/bin/env python3
"""
使用固定单个样本对比两个仓库的数据处理逻辑（完整版）

此脚本直接使用 Dataset 类处理样本，确保输出完整的训练数据字段：
- input_ids, position_ids, loss_mask
- pixel_values, pixel_values_videos
- image_grid_thw, video_grid_thw
- fast_pixel_values_videos, fast_video_grid_thw
- cu_seqlens, sample_idx, epoch_idx

使用方法:
    # 在 end2end 目录下运行:
    cd /llm_reco/maosiyang/video_tok/muse
    python verify_dataset_real_output.py --mode end2end
    
    # 在 msy_master_2 目录下运行:
    cd /llm_reco/maosiyang/msy_master_2/muse
    python verify_dataset_real_output.py --mode msy_master_2
    
    # 对比结果:
    python verify_dataset_real_output.py --mode compare
"""

import argparse
import json
import os
import sys
import numpy as np

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
MODEL_PATH = "/llm_reco_ssd/maosiyang/models/muse/keye_tokenizer_video_baseline"
MODEL_PATH_2 = "/llm_reco_ssd/maosiyang/models/muse/keye_tokenizer_end2end_image_for_stage_2_video"

# 输出目录
OUTPUT_DIR = "/llm_reco/maosiyang/temp"


def save_batch_summary(batch, output_file, text_prompt=None, extra_info=None):
    """保存 batch 的摘要信息（用于跨环境对比）"""
    summary = {
        "keys": list(batch.keys()),
    }
    
    if text_prompt:
        summary["text_prompt"] = text_prompt
    
    if extra_info:
        summary["extra_info"] = extra_info
    
    for key, value in batch.items():
        if value is None:
            summary[f"{key}"] = None
            continue
            
        if hasattr(value, 'shape'):
            summary[f"{key}_shape"] = list(value.shape)
            summary[f"{key}_dtype"] = str(value.dtype)
            if hasattr(value, 'numpy'):
                arr = value.cpu().numpy() if hasattr(value, 'cpu') else np.array(value)
            else:
                arr = np.array(value)
            summary[f"{key}_sum"] = float(arr.sum())
            summary[f"{key}_mean"] = float(arr.mean()) if arr.size > 0 else 0
            summary[f"{key}_min"] = float(arr.min()) if arr.size > 0 else 0
            summary[f"{key}_max"] = float(arr.max()) if arr.size > 0 else 0
            summary[f"{key}_numel"] = int(arr.size)
            # 保存前500个元素用于精确对比
            flat = arr.flatten()[:500]
            summary[f"{key}_first500"] = flat.tolist()
            # 保存全部数据用于详细对比（限制大小）
            flat_all = arr.flatten()
            if len(flat_all) <= 50000:  # 限制保存的最大元素数
                summary[f"{key}_all"] = flat_all.tolist()
            else:
                summary[f"{key}_all_truncated"] = True
                summary[f"{key}_all"] = flat_all[:50000].tolist()
        elif isinstance(value, (list, tuple)):
            summary[f"{key}_len"] = len(value)
            summary[f"{key}_content"] = str(value)[:2000]
        else:
            summary[f"{key}"] = str(value)[:2000]
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"Saved batch summary to {output_file}")
    return summary


def run_end2end():
    """在 end2end 环境下运行 - 使用完整 Dataset 处理流程"""
    print("=" * 70)
    print("Running in end2end mode - Full Dataset Processing")
    print("=" * 70)
    
    # 添加路径
    sys.path.insert(0, os.getcwd())
    
    # 设置环境变量
    os.environ['nosp'] = '1'
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['LOCAL_RANK'] = '0'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    
    import torch
    import torch.distributed as dist
    
    # 初始化分布式环境（需要用于某些 dataset 操作）
    if not dist.is_initialized():
        dist.init_process_group(backend='gloo', init_method='env://')
    
    from transformers import AutoProcessor
    
    # 导入 process_vision_info
    from recovlm.models.tokenizer_end2end_mt_1drope_video.keye_vl_utils import process_vision_info
    
    try:
        print(f"\n📁 Model path: {MODEL_PATH}")
        print(f"📹 Video path: {TEST_VIDEO_PATH}")
        print(f"💬 Messages: {json.dumps(TEST_MESSAGES, indent=2, ensure_ascii=False)}")
        
        # 加载 processor
        print("\n⚙️ Loading Processor...")
        processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
        
        # 应用 chat template
        print("\n📝 Applying Chat Template...")
        text = processor.apply_chat_template(
            TEST_MESSAGES,
            tokenize=False,
            add_generation_prompt=False
        )
        text += "<|endoftext|>"
        print(f"   -> Text: {repr(text[:200])}...")
        
        # 提取视觉信息
        print("\n🎥 Extracting Vision Info...")
        image_inputs, video_inputs = process_vision_info(TEST_MESSAGES)
        print(f"   -> Images: {len(image_inputs) if image_inputs else 0}")
        print(f"   -> Videos: {len(video_inputs) if video_inputs else 0}")
        
        # Step 1: 使用 Processor 获取基础输入
        print("\n🔄 Step 1: Running Processor...")
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            truncation=False,
            return_tensors="pt",
        )
        
        print(f"   Processor output keys: {list(inputs.keys())}")
        
        # Step 2: 计算 position_ids（使用 end2end 的方法）
        print("\n🔄 Step 2: Computing position_ids...")
        try:
            from recovlm.models.tokenizer_end2end_mt_1drope_video.modeling import get_rope_index_slowfast_video_tok_3d
        except ImportError:
            from recovlm.data.datasets import get_rope_index_slowfast_video_tok_3d
        
        position_ids = get_rope_index_slowfast_video_tok_3d(
            input_ids=inputs['input_ids'],
            image_grid_thw=inputs.get('image_grid_thw'),
            video_grid_thw=inputs.get('video_grid_thw'),
            fast_video_grid_thw=inputs.get('fast_video_grid_thw'),
            attention_mask=inputs.get('attention_mask'),
            image_token_id=processor.image_token_id if hasattr(processor, 'image_token_id') else 151655,
            video_token_id=processor.video_token_id if hasattr(processor, 'video_token_id') else 151656,
            fast_video_token_id=getattr(processor, 'fast_video_token_id', 151678),
            spatial_merge_size=2,
            vision_start_token_id=151652,
        )
        inputs['position_ids'] = position_ids
        print(f"   position_ids shape: {position_ids.shape}")
        
        # Step 3: 计算 loss_mask
        print("\n🔄 Step 3: Computing loss_mask...")
        # loss_mask: 只对 assistant 回复部分计算 loss
        input_ids = inputs['input_ids'].flatten()
        loss_mask = torch.ones_like(input_ids, dtype=torch.float32)
        
        # 找到 assistant 回复开始和结束的位置
        # 通常格式: <|im_start|>assistant\n...content...<|im_end|><|endoftext|>
        # im_start_id = 151644, im_end_id = 151645
        im_start_id = 151644
        im_end_id = 151645
        
        # 简单策略：不计算 vision tokens 的 loss
        vision_start_id = 151652
        vision_end_id = 151653
        
        # 设置 vision tokens 的 loss_mask 为 0
        in_vision = False
        for i in range(len(input_ids)):
            if input_ids[i] == vision_start_id:
                in_vision = True
            if in_vision:
                loss_mask[i] = 0
            if input_ids[i] == vision_end_id:
                in_vision = False
        
        inputs['loss_mask'] = loss_mask.unsqueeze(0)
        print(f"   loss_mask shape: {inputs['loss_mask'].shape}")
        print(f"   loss_mask sum: {inputs['loss_mask'].sum().item()}")
        
        # Step 4: 添加其他训练字段
        print("\n🔄 Step 4: Adding training fields...")
        inputs['cu_seqlens'] = torch.tensor([0, inputs['input_ids'].shape[1]], dtype=torch.int32)
        inputs['sample_idx'] = torch.zeros_like(inputs['input_ids'], dtype=torch.int32)
        inputs['epoch_idx'] = torch.tensor([0.0], dtype=torch.float32)
        
        # 确保有 fast_video 相关字段
        if 'fast_pixel_values_videos' not in inputs:
            inputs['fast_pixel_values_videos'] = None
        if 'fast_video_grid_thw' not in inputs:
            inputs['fast_video_grid_thw'] = None
        
        # 打印所有字段
        print(f"\n✅ Final output keys: {list(inputs.keys())}")
        for key, value in inputs.items():
            if value is None:
                print(f"   {key}: None")
            elif hasattr(value, 'shape'):
                print(f"   {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"   {key}: {type(value)}")
        
        # 额外信息
        extra_info = {
            "model_path": MODEL_PATH,
            "video_path": TEST_VIDEO_PATH,
            "messages": TEST_MESSAGES,
        }
        
        # 保存摘要
        output_file = os.path.join(OUTPUT_DIR, "batch_summary_end2end.json")
        save_batch_summary(inputs, output_file, text_prompt=text, extra_info=extra_info)
        
        # 详细打印关键信息
        print("\n📊 Detailed tensor values:")
        if 'input_ids' in inputs:
            ids = inputs['input_ids'].flatten().tolist()
            print(f"   input_ids length: {len(ids)}")
            print(f"   input_ids first 50: {ids[:50]}")
            print(f"   input_ids last 20: {ids[-20:]}")
        
        if 'position_ids' in inputs and inputs['position_ids'] is not None:
            pos = inputs['position_ids']
            print(f"   position_ids shape: {pos.shape}")
            if len(pos.shape) == 3:  # [3, 1, seq_len]
                print(f"   position_ids[0] first 20: {pos[0, 0, :20].tolist()}")
                print(f"   position_ids[1] first 20: {pos[1, 0, :20].tolist()}")
                print(f"   position_ids[2] first 20: {pos[2, 0, :20].tolist()}")
            else:
                print(f"   position_ids first 20: {pos.flatten()[:20].tolist()}")
        
        if 'loss_mask' in inputs:
            lm = inputs['loss_mask'].flatten()
            print(f"   loss_mask sum: {lm.sum().item()}, total: {len(lm)}")
            print(f"   loss_mask first 50: {lm[:50].tolist()}")
        
        if 'video_grid_thw' in inputs and inputs['video_grid_thw'] is not None:
            print(f"   video_grid_thw: {inputs['video_grid_thw'].tolist()}")
        
        if 'fast_video_grid_thw' in inputs and inputs['fast_video_grid_thw'] is not None:
            print(f"   fast_video_grid_thw: {inputs['fast_video_grid_thw'].tolist()}")
        
        if 'cu_seqlens' in inputs:
            print(f"   cu_seqlens: {inputs['cu_seqlens'].tolist()}")
            
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()


def run_msy_master_2():
    """在 msy_master_2 环境下运行 - 使用完整 Dataset 处理流程"""
    print("=" * 70)
    print("Running in msy_master_2 mode - Full Dataset Processing")
    print("=" * 70)
    
    # 添加路径
    sys.path.insert(0, os.getcwd())
    
    # 设置环境变量
    os.environ['nosp'] = '1'
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['LOCAL_RANK'] = '0'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29501'
    
    import torch
    import torch.distributed as dist
    
    # 初始化分布式环境
    if not dist.is_initialized():
        dist.init_process_group(backend='gloo', init_method='env://')
    
    from transformers import AutoProcessor
    
    # 导入 process_vision_info
    try:
        from tests.models.tokenizer_end2end_mt_1drope_video.keye_vl_utils import process_vision_info
    except ImportError:
        from muse.models.keye_tokenizer_end2end_video.keye_vl_utils import process_vision_info
    
    try:
        print(f"\n📁 Model path: {MODEL_PATH_2}")
        print(f"📹 Video path: {TEST_VIDEO_PATH}")
        print(f"💬 Messages: {json.dumps(TEST_MESSAGES, indent=2, ensure_ascii=False)}")
        
        # 加载 processor
        print("\n⚙️ Loading Processor...")
        processor = AutoProcessor.from_pretrained(MODEL_PATH_2, trust_remote_code=True)
        
        # 应用 chat template
        print("\n📝 Applying Chat Template...")
        text = processor.apply_chat_template(
            TEST_MESSAGES,
            tokenize=False,
            add_generation_prompt=False
        )
        text += "<|endoftext|>"
        print(f"   -> Text: {repr(text[:200])}...")
        
        # 提取视觉信息
        print("\n🎥 Extracting Vision Info...")
        image_inputs, video_inputs = process_vision_info(TEST_MESSAGES)
        print(f"   -> Images: {len(image_inputs) if image_inputs else 0}")
        print(f"   -> Videos: {len(video_inputs) if video_inputs else 0}")
        
        # Step 1: 使用 Processor 获取基础输入
        print("\n🔄 Step 1: Running Processor...")
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            truncation=False,
            return_tensors="pt",
        )
        
        print(f"   Processor output keys: {list(inputs.keys())}")
        
        # Step 2: 计算 position_ids（使用 msy_master_2 的方法）
        print("\n🔄 Step 2: Computing position_ids...")
        from muse.data.datasets.tokenizer_dataset_video import get_rope_index_slowfast_video_tok_3d
        
        position_ids = get_rope_index_slowfast_video_tok_3d(
            input_ids=inputs['input_ids'],
            image_grid_thw=inputs.get('image_grid_thw'),
            video_grid_thw=inputs.get('video_grid_thw'),
            fast_video_grid_thw=inputs.get('fast_video_grid_thw'),
            attention_mask=inputs.get('attention_mask'),
            image_token_id=processor.image_token_id if hasattr(processor, 'image_token_id') else 151655,
            video_token_id=processor.video_token_id if hasattr(processor, 'video_token_id') else 151656,
            fast_video_token_id=getattr(processor, 'fast_video_token_id', 151678),
            spatial_merge_size=2,
            vision_start_token_id=151652,
        )
        inputs['position_ids'] = position_ids
        print(f"   position_ids shape: {position_ids.shape}")
        
        # Step 3: 计算 loss_mask
        print("\n🔄 Step 3: Computing loss_mask...")
        input_ids = inputs['input_ids'].flatten()
        loss_mask = torch.ones_like(input_ids, dtype=torch.float32)
        
        # 设置 vision tokens 的 loss_mask 为 0
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
        print(f"   loss_mask shape: {inputs['loss_mask'].shape}")
        print(f"   loss_mask sum: {inputs['loss_mask'].sum().item()}")
        
        # Step 4: 添加其他训练字段
        print("\n🔄 Step 4: Adding training fields...")
        inputs['cu_seqlens'] = torch.tensor([0, inputs['input_ids'].shape[1]], dtype=torch.int32)
        inputs['sample_idx'] = torch.zeros_like(inputs['input_ids'], dtype=torch.int32)
        inputs['epoch_idx'] = torch.tensor([0.0], dtype=torch.float32)
        
        # 确保有 fast_video 相关字段
        if 'fast_pixel_values_videos' not in inputs:
            inputs['fast_pixel_values_videos'] = None
        if 'fast_video_grid_thw' not in inputs:
            inputs['fast_video_grid_thw'] = None
        
        # 打印所有字段
        print(f"\n✅ Final output keys: {list(inputs.keys())}")
        for key, value in inputs.items():
            if value is None:
                print(f"   {key}: None")
            elif hasattr(value, 'shape'):
                print(f"   {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"   {key}: {type(value)}")
        
        # 额外信息
        extra_info = {
            "model_path": MODEL_PATH_2,
            "video_path": TEST_VIDEO_PATH,
            "messages": TEST_MESSAGES,
        }
        
        # 保存摘要
        output_file = os.path.join(OUTPUT_DIR, "batch_summary_msy_master_2.json")
        save_batch_summary(inputs, output_file, text_prompt=text, extra_info=extra_info)
        
        # 详细打印关键信息
        print("\n📊 Detailed tensor values:")
        if 'input_ids' in inputs:
            ids = inputs['input_ids'].flatten().tolist()
            print(f"   input_ids length: {len(ids)}")
            print(f"   input_ids first 50: {ids[:50]}")
            print(f"   input_ids last 20: {ids[-20:]}")
        
        if 'position_ids' in inputs and inputs['position_ids'] is not None:
            pos = inputs['position_ids']
            print(f"   position_ids shape: {pos.shape}")
            if len(pos.shape) == 3:  # [3, 1, seq_len]
                print(f"   position_ids[0] first 20: {pos[0, 0, :20].tolist()}")
                print(f"   position_ids[1] first 20: {pos[1, 0, :20].tolist()}")
                print(f"   position_ids[2] first 20: {pos[2, 0, :20].tolist()}")
            else:
                print(f"   position_ids first 20: {pos.flatten()[:20].tolist()}")
        
        if 'loss_mask' in inputs:
            lm = inputs['loss_mask'].flatten()
            print(f"   loss_mask sum: {lm.sum().item()}, total: {len(lm)}")
            print(f"   loss_mask first 50: {lm[:50].tolist()}")
        
        if 'video_grid_thw' in inputs and inputs['video_grid_thw'] is not None:
            print(f"   video_grid_thw: {inputs['video_grid_thw'].tolist()}")
        
        if 'fast_video_grid_thw' in inputs and inputs['fast_video_grid_thw'] is not None:
            print(f"   fast_video_grid_thw: {inputs['fast_video_grid_thw'].tolist()}")
        
        if 'cu_seqlens' in inputs:
            print(f"   cu_seqlens: {inputs['cu_seqlens'].tolist()}")
            
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()


def compare_summaries():
    """详细对比两个摘要文件的所有字段"""
    print("=" * 70)
    print("Comparing batch summaries (Full Dataset Output)")
    print("=" * 70)
    
    end2end_file = os.path.join(OUTPUT_DIR, "batch_summary_end2end.json")
    msy_file = os.path.join(OUTPUT_DIR, "batch_summary_msy_master_2.json")
    
    if not os.path.exists(end2end_file):
        print(f"Missing: {end2end_file}")
        print("Please run: cd end2end && python verify_dataset_real_output.py --mode end2end")
        return
    
    if not os.path.exists(msy_file):
        print(f"Missing: {msy_file}")
        print("Please run: cd msy_master_2 && python verify_dataset_real_output.py --mode msy_master_2")
        return
    
    with open(end2end_file, 'r', encoding='utf-8') as f:
        end2end = json.load(f)
    
    with open(msy_file, 'r', encoding='utf-8') as f:
        msy = json.load(f)
    
    differences_found = []
    
    # 1. 对比 text prompt
    print("\n" + "=" * 50)
    print("## 1. Text Prompt Comparison")
    print("=" * 50)
    e2e_text = end2end.get('text_prompt', '')
    msy_text = msy.get('text_prompt', '')
    if e2e_text == msy_text:
        print(f"  ✅ Text prompts are IDENTICAL")
        print(f"     Length: {len(e2e_text)}")
    else:
        print(f"  ❌ Text prompts DIFFER!")
        print(f"     end2end length: {len(e2e_text)}")
        print(f"     msy length: {len(msy_text)}")
        print(f"     end2end: {repr(e2e_text[:300])}")
        print(f"     msy:     {repr(msy_text[:300])}")
        differences_found.append("text_prompt")
    
    # 2. 对比 keys
    print("\n" + "=" * 50)
    print("## 2. Keys Comparison")
    print("=" * 50)
    end2end_keys = set(end2end.get('keys', []))
    msy_keys = set(msy.get('keys', []))
    
    common_keys = end2end_keys & msy_keys
    only_end2end = end2end_keys - msy_keys
    only_msy = msy_keys - end2end_keys
    
    print(f"  Common keys ({len(common_keys)}): {sorted(common_keys)}")
    if only_end2end:
        print(f"  ⚠️ Only in end2end: {only_end2end}")
        differences_found.append(f"keys_only_end2end: {only_end2end}")
    if only_msy:
        print(f"  ⚠️ Only in msy_master_2: {only_msy}")
        differences_found.append(f"keys_only_msy: {only_msy}")
    
    # 3. 对比 shapes
    print("\n" + "=" * 50)
    print("## 3. Shape Comparison")
    print("=" * 50)
    shape_matches = 0
    shape_mismatches = 0
    for key in sorted(common_keys):
        shape_key = f"{key}_shape"
        if shape_key in end2end and shape_key in msy:
            e2e_shape = end2end[shape_key]
            msy_shape = msy[shape_key]
            if e2e_shape == msy_shape:
                print(f"  ✅ {key}: {e2e_shape}")
                shape_matches += 1
            else:
                print(f"  ❌ {key}: end2end={e2e_shape}, msy={msy_shape}")
                shape_mismatches += 1
                differences_found.append(f"shape_{key}")
        elif shape_key in end2end:
            print(f"  ⚠️ {key}: end2end={end2end[shape_key]}, msy=N/A")
        elif shape_key in msy:
            print(f"  ⚠️ {key}: end2end=N/A, msy={msy[shape_key]}")
    
    print(f"\n  Summary: {shape_matches} matches, {shape_mismatches} mismatches")
    
    # 4. 对比数值统计
    print("\n" + "=" * 50)
    print("## 4. Value Statistics Comparison")
    print("=" * 50)
    
    stat_keys = ['sum', 'mean', 'min', 'max', 'numel']
    for key in sorted(common_keys):
        has_stats = any(f"{key}_{s}" in end2end for s in stat_keys)
        if not has_stats:
            continue
        
        print(f"\n  {key}:")
        for stat in stat_keys:
            stat_key = f"{key}_{stat}"
            if stat_key in end2end and stat_key in msy:
                e2e_val = end2end[stat_key]
                msy_val = msy[stat_key]
                if isinstance(e2e_val, (int, float)) and isinstance(msy_val, (int, float)):
                    diff = abs(e2e_val - msy_val)
                    rel_diff = diff / max(abs(e2e_val), abs(msy_val), 1e-10)
                    if rel_diff < 1e-5:
                        print(f"    ✅ {stat}: {e2e_val:.6g}")
                    else:
                        print(f"    ❌ {stat}: end2end={e2e_val:.6g}, msy={msy_val:.6g}, diff={diff:.6g}")
                        if stat == 'sum' or stat == 'mean':
                            differences_found.append(f"value_{key}_{stat}")
    
    # 5. 详细对比关键字段的值
    print("\n" + "=" * 50)
    print("## 5. Detailed Value Comparison (Critical Fields)")
    print("=" * 50)
    
    critical_fields = [
        'input_ids', 'position_ids', 'loss_mask',
        'pixel_values', 'pixel_values_videos',
        'image_grid_thw', 'video_grid_thw',
        'fast_pixel_values_videos', 'fast_video_grid_thw',
        'cu_seqlens', 'sample_idx'
    ]
    
    for key in critical_fields:
        if key not in common_keys:
            # 检查是否一方为 None
            e2e_none = end2end.get(f"{key}") is None or end2end.get(f"{key}_shape") is None
            msy_none = msy.get(f"{key}") is None or msy.get(f"{key}_shape") is None
            if e2e_none and msy_none:
                print(f"\n  {key}: Both None ✅")
            elif e2e_none:
                print(f"\n  {key}: end2end=None, msy has value ⚠️")
            elif msy_none:
                print(f"\n  {key}: end2end has value, msy=None ⚠️")
            continue
        
        all_key = f"{key}_all"
        first500_key = f"{key}_first500"
        
        # 优先使用 all 数据，否则使用 first500
        if all_key in end2end and all_key in msy:
            e2e_vals = np.array(end2end[all_key])
            msy_vals = np.array(msy[all_key])
            data_source = "all"
        elif first500_key in end2end and first500_key in msy:
            e2e_vals = np.array(end2end[first500_key])
            msy_vals = np.array(msy[first500_key])
            data_source = "first500"
        else:
            print(f"\n  {key}: No data available for comparison")
            continue
        
        print(f"\n  {key} (using {data_source} data):")
        print(f"    end2end length: {len(e2e_vals)}")
        print(f"    msy length: {len(msy_vals)}")
        
        if len(e2e_vals) != len(msy_vals):
            print(f"    ❌ Length mismatch!")
            differences_found.append(f"length_{key}")
            min_len = min(len(e2e_vals), len(msy_vals))
            e2e_vals = e2e_vals[:min_len]
            msy_vals = msy_vals[:min_len]
        
        if len(e2e_vals) == 0:
            print(f"    (empty)")
            continue
        
        # 精确匹配（对于整数类型）
        if key in ['input_ids', 'cu_seqlens', 'sample_idx']:
            exact_matches = np.sum(e2e_vals == msy_vals)
            total = len(e2e_vals)
            if exact_matches == total:
                print(f"    ✅ Exact match: {exact_matches}/{total}")
            else:
                print(f"    ❌ Exact match: {exact_matches}/{total}")
                differences_found.append(f"exact_{key}")
                # 找出第一个不匹配
                diff_idx = np.where(e2e_vals != msy_vals)[0]
                print(f"    First 10 diff indices: {diff_idx[:10].tolist()}")
                for idx in diff_idx[:5]:
                    print(f"      idx {idx}: e2e={int(e2e_vals[idx])}, msy={int(msy_vals[idx])}")
        else:
            # 近似匹配（对于浮点类型）
            close_matches = np.sum(np.isclose(e2e_vals, msy_vals, rtol=1e-5, atol=1e-5))
            total = len(e2e_vals)
            diff = np.abs(e2e_vals - msy_vals)
            
            if close_matches == total:
                print(f"    ✅ Close match: {close_matches}/{total}")
            else:
                print(f"    ❌ Close match: {close_matches}/{total}")
                print(f"    Max diff: {diff.max():.6e}")
                print(f"    Mean diff: {diff.mean():.6e}")
                differences_found.append(f"close_{key}")
                
                mismatch_idx = np.where(~np.isclose(e2e_vals, msy_vals, rtol=1e-5, atol=1e-5))[0]
                if len(mismatch_idx) > 0:
                    print(f"    First 10 mismatch indices: {mismatch_idx[:10].tolist()}")
                    for idx in mismatch_idx[:3]:
                        print(f"      idx {idx}: e2e={e2e_vals[idx]:.6f}, msy={msy_vals[idx]:.6f}")
    
    # 6. position_ids 特别对比（3D RoPE）
    print("\n" + "=" * 50)
    print("## 6. Position IDs Detailed Comparison (3D RoPE)")
    print("=" * 50)
    
    pos_key = "position_ids_all"
    if pos_key in end2end and pos_key in msy:
        e2e_pos = np.array(end2end[pos_key])
        msy_pos = np.array(msy[pos_key])
        
        e2e_shape = end2end.get("position_ids_shape", [])
        msy_shape = msy.get("position_ids_shape", [])
        
        print(f"  end2end shape: {e2e_shape}")
        print(f"  msy shape: {msy_shape}")
        
        if e2e_shape == msy_shape and len(e2e_pos) == len(msy_pos):
            # 重塑为原始形状
            if len(e2e_shape) == 3 and e2e_shape[0] == 3:
                seq_len = e2e_shape[2]
                try:
                    e2e_pos_reshaped = e2e_pos.reshape(e2e_shape)
                    msy_pos_reshaped = msy_pos.reshape(msy_shape)
                    
                    for dim in range(3):
                        dim_name = ['temporal', 'height', 'width'][dim]
                        e2e_dim = e2e_pos_reshaped[dim].flatten()
                        msy_dim = msy_pos_reshaped[dim].flatten()
                        matches = np.sum(e2e_dim == msy_dim)
                        if matches == len(e2e_dim):
                            print(f"  ✅ {dim_name} dimension: {matches}/{len(e2e_dim)} match")
                        else:
                            print(f"  ❌ {dim_name} dimension: {matches}/{len(e2e_dim)} match")
                            diff_idx = np.where(e2e_dim != msy_dim)[0]
                            print(f"     First diffs at: {diff_idx[:5].tolist()}")
                except Exception as e:
                    print(f"  Error reshaping: {e}")
    
    # 7. 总结
    print("\n" + "=" * 70)
    print("## SUMMARY")
    print("=" * 70)
    
    if len(differences_found) == 0:
        print("🎉 ALL CHECKS PASSED! The two datasets produce identical output.")
    else:
        print(f"⚠️ Found {len(differences_found)} differences:")
        for diff in differences_found:
            print(f"   - {diff}")
        print("\nThese differences may explain the training behavior discrepancy.")


def main():
    parser = argparse.ArgumentParser(description='Verify dataset output equivalence with full training fields')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['end2end', 'msy_master_2', 'compare'],
                        help='Run mode')
    args = parser.parse_args()
    
    if args.mode == 'end2end':
        run_end2end()
    elif args.mode == 'msy_master_2':
        run_msy_master_2()
    elif args.mode == 'compare':
        compare_summaries()


if __name__ == "__main__":
    main()
