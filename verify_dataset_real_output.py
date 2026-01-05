#!/usr/bin/env python3
"""
使用固定单个样本对比两个仓库的数据处理逻辑

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

def save_batch_summary(batch, output_file, text_prompt=None):
    """保存 batch 的摘要信息（用于跨环境对比）"""
    summary = {
        "keys": list(batch.keys()),
    }
    
    if text_prompt:
        summary["text_prompt"] = text_prompt
    
    for key, value in batch.items():
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
            # 保存前500个元素用于精确对比
            flat = arr.flatten()[:500]
            summary[f"{key}_first500"] = flat.tolist()
            # 保存全部数据用于详细对比
            summary[f"{key}_all"] = arr.flatten().tolist()
        elif isinstance(value, (list, tuple)):
            summary[f"{key}_len"] = len(value)
            summary[f"{key}_content"] = str(value)[:1000]
        elif value is None:
            summary[f"{key}"] = None
        else:
            summary[f"{key}"] = str(value)[:1000]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"Saved batch summary to {output_file}")
    return summary


def run_end2end():
    """在 end2end 环境下运行 - 使用 Processor 处理固定样本"""
    print("=" * 70)
    print("Running in end2end mode - Single Sample Test")
    print("=" * 70)
    
    # 添加路径
    sys.path.insert(0, os.getcwd())
    
    # 设置环境变量
    os.environ['nosp'] = '1'
    
    import torch
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
            add_generation_prompt=False  # 因为已有 assistant 回答
        )
        # 添加 EOS token
        text += "<|endoftext|>"
        print(f"   -> Text: {repr(text[:200])}...")
        
        # 提取视觉信息
        print("\n🎥 Extracting Vision Info...")
        image_inputs, video_inputs = process_vision_info(TEST_MESSAGES)
        print(f"   -> Images: {len(image_inputs) if image_inputs else 0}")
        print(f"   -> Videos: {len(video_inputs) if video_inputs else 0}")
        
        # 运行 processor
        print("\n🔄 Running Processor...")
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            truncation=False,
            return_tensors="pt",
        )
        
        print(f"\n✅ Processor output keys: {list(inputs.keys())}")
        for key, value in inputs.items():
            if hasattr(value, 'shape'):
                print(f"   {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"   {key}: {type(value)}")
        
        # 保存摘要
        output_file = "/llm_reco/maosiyang/temp/batch_summary_end2end.json"
        save_batch_summary(inputs, output_file, text_prompt=text)
        
        # 额外打印一些关键信息
        print("\n📊 Key tensor values:")
        if 'input_ids' in inputs:
            ids = inputs['input_ids'].flatten().tolist()
            print(f"   input_ids length: {len(ids)}")
            print(f"   input_ids first 50: {ids[:50]}")
            print(f"   input_ids last 20: {ids[-20:]}")
        
        if 'video_grid_thw' in inputs:
            print(f"   video_grid_thw: {inputs['video_grid_thw'].tolist()}")
        
        if 'image_grid_thw' in inputs:
            print(f"   image_grid_thw: {inputs['image_grid_thw'].tolist()}")
            
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()


def run_msy_master_2():
    """在 msy_master_2 环境下运行 - 使用 Processor 处理固定样本"""
    print("=" * 70)
    print("Running in msy_master_2 mode - Single Sample Test")
    print("=" * 70)
    
    # 添加路径
    sys.path.insert(0, os.getcwd())
    
    # 设置环境变量
    os.environ['nosp'] = '1'
    
    import torch
    from transformers import AutoProcessor
    
    # 导入 process_vision_info - msy_master_2 的路径可能不同
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
        # 添加 EOS token
        text += "<|endoftext|>"
        print(f"   -> Text: {repr(text[:200])}...")
        
        # 提取视觉信息
        print("\n🎥 Extracting Vision Info...")
        image_inputs, video_inputs = process_vision_info(TEST_MESSAGES)
        print(f"   -> Images: {len(image_inputs) if image_inputs else 0}")
        print(f"   -> Videos: {len(video_inputs) if video_inputs else 0}")
        
        # 运行 processor
        print("\n🔄 Running Processor...")
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            truncation=False,
            return_tensors="pt",
        )
        
        print(f"\n✅ Processor output keys: {list(inputs.keys())}")
        for key, value in inputs.items():
            if hasattr(value, 'shape'):
                print(f"   {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"   {key}: {type(value)}")
        
        # 保存摘要
        output_file = "/llm_reco/maosiyang/temp/batch_summary_msy_master_2.json"
        save_batch_summary(inputs, output_file, text_prompt=text)
        
        # 额外打印一些关键信息
        print("\n📊 Key tensor values:")
        if 'input_ids' in inputs:
            ids = inputs['input_ids'].flatten().tolist()
            print(f"   input_ids length: {len(ids)}")
            print(f"   input_ids first 50: {ids[:50]}")
            print(f"   input_ids last 20: {ids[-20:]}")
        
        if 'video_grid_thw' in inputs:
            print(f"   video_grid_thw: {inputs['video_grid_thw'].tolist()}")
        
        if 'image_grid_thw' in inputs:
            print(f"   image_grid_thw: {inputs['image_grid_thw'].tolist()}")
            
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()


def compare_summaries():
    """对比两个摘要文件"""
    print("=" * 70)
    print("Comparing batch summaries (Single Sample)")
    print("=" * 70)
    
    end2end_file = "/llm_reco/maosiyang/temp/batch_summary_end2end.json"
    msy_file = "/llm_reco/maosiyang/temp/batch_summary_msy_master_2.json"
    
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
    
    # 对比 text prompt
    print("\n## Text Prompt Comparison:")
    e2e_text = end2end.get('text_prompt', '')
    msy_text = msy.get('text_prompt', '')
    if e2e_text == msy_text:
        print(f"  ✅ Text prompts are identical")
        print(f"     Length: {len(e2e_text)}")
    else:
        print(f"  ❌ Text prompts differ!")
        print(f"     end2end: {repr(e2e_text[:200])}")
        print(f"     msy:     {repr(msy_text[:200])}")
    
    print("\n## Keys comparison:")
    end2end_keys = set(end2end.get('keys', []))
    msy_keys = set(msy.get('keys', []))
    
    common_keys = end2end_keys & msy_keys
    only_end2end = end2end_keys - msy_keys
    only_msy = msy_keys - end2end_keys
    
    print(f"  Common keys: {sorted(common_keys)}")
    if only_end2end:
        print(f"  Only in end2end: {only_end2end}")
    if only_msy:
        print(f"  Only in msy_master_2: {only_msy}")
    
    print("\n## Shape comparison for common keys:")
    all_match = True
    for key in sorted(common_keys):
        shape_key = f"{key}_shape"
        if shape_key in end2end and shape_key in msy:
            e2e_shape = end2end[shape_key]
            msy_shape = msy[shape_key]
            match = "✅" if e2e_shape == msy_shape else "❌"
            if e2e_shape != msy_shape:
                all_match = False
            print(f"  {key}: end2end={e2e_shape}, msy={msy_shape} {match}")
    
    if all_match:
        print("\n  🎉 All shapes match!")
    
    print("\n## Value comparison for critical fields:")
    critical_fields = ['input_ids', 'pixel_values_videos', 'video_grid_thw', 'image_grid_thw']
    
    for key in critical_fields:
        if key not in common_keys:
            continue
            
        first_key = f"{key}_first500"
        if first_key in end2end and first_key in msy:
            e2e_vals = np.array(end2end[first_key])
            msy_vals = np.array(msy[first_key])
            
            n = min(len(e2e_vals), len(msy_vals))
            if n == 0:
                continue
                
            e2e_slice = e2e_vals[:n]
            msy_slice = msy_vals[:n]
            
            match_count = np.sum(np.isclose(e2e_slice, msy_slice, rtol=1e-5, atol=1e-5))
            diff = np.abs(e2e_slice - msy_slice)
            
            status = "✅" if match_count == n else "❌"
            print(f"\n  {key}: {status}")
            print(f"    Match: {match_count}/{n}")
            if match_count < n:
                print(f"    Max diff: {diff.max():.6f}")
                print(f"    Mean diff: {diff.mean():.6f}")
                # 找出第一个不匹配的位置
                mismatch_idx = np.where(~np.isclose(e2e_slice, msy_slice, rtol=1e-5, atol=1e-5))[0]
                if len(mismatch_idx) > 0:
                    first_mismatch = mismatch_idx[0]
                    print(f"    First mismatch at index {first_mismatch}: e2e={e2e_slice[first_mismatch]}, msy={msy_slice[first_mismatch]}")
    
    # 详细对比 input_ids
    print("\n## Detailed input_ids comparison:")
    e2e_ids = end2end.get('input_ids_all', end2end.get('input_ids_first500', []))
    msy_ids = msy.get('input_ids_all', msy.get('input_ids_first500', []))
    
    if e2e_ids and msy_ids:
        e2e_ids = np.array(e2e_ids).flatten()
        msy_ids = np.array(msy_ids).flatten()
        
        print(f"  end2end length: {len(e2e_ids)}")
        print(f"  msy length: {len(msy_ids)}")
        
        if len(e2e_ids) == len(msy_ids):
            matches = np.sum(e2e_ids == msy_ids)
            print(f"  Exact matches: {matches}/{len(e2e_ids)}")
            if matches == len(e2e_ids):
                print("  🎉 input_ids are IDENTICAL!")
            else:
                # 找出不同的位置
                diff_idx = np.where(e2e_ids != msy_ids)[0]
                print(f"  Differences at indices: {diff_idx[:20].tolist()}...")
                for idx in diff_idx[:5]:
                    print(f"    idx {idx}: e2e={int(e2e_ids[idx])}, msy={int(msy_ids[idx])}")
        else:
            print(f"  ❌ Length mismatch!")
            # 找到共同前缀长度
            min_len = min(len(e2e_ids), len(msy_ids))
            for i in range(min_len):
                if e2e_ids[i] != msy_ids[i]:
                    print(f"  First difference at index {i}: e2e={int(e2e_ids[i])}, msy={int(msy_ids[i])}")
                    break
            else:
                print(f"  First {min_len} elements are identical")


def main():
    parser = argparse.ArgumentParser(description='Verify dataset output equivalence with single sample')
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
