#!/usr/bin/env python3
"""
实际加载数据集，对比两个仓库的数据输出差异

使用方法:
    # 在 end2end 目录下运行:
    cd /Users/maosiyang/Desktop/temp/end2end/muse
    python ../../verify_dataset_real_output.py --mode end2end
    
    # 在 msy_master_2 目录下运行:
    cd /Users/maosiyang/Desktop/temp/msy_master_2/muse
    python ../../verify_dataset_real_output.py --mode msy_master_2
    
    # 对比结果:
    python /Users/maosiyang/Desktop/temp/verify_dataset_real_output.py --mode compare
"""

import argparse
import json
import os
import pickle
import sys
import numpy as np

def save_batch_summary(batch, output_file):
    """保存 batch 的摘要信息（用于跨环境对比）"""
    summary = {
        "keys": list(batch.keys()),
    }
    
    for key, value in batch.items():
        if hasattr(value, 'shape'):
            summary[f"{key}_shape"] = list(value.shape)
            summary[f"{key}_dtype"] = str(value.dtype)
            if hasattr(value, 'numpy'):
                arr = value.numpy() if hasattr(value, 'numpy') else np.array(value)
                summary[f"{key}_sum"] = float(arr.sum())
                summary[f"{key}_mean"] = float(arr.mean()) if arr.size > 0 else 0
                summary[f"{key}_min"] = float(arr.min()) if arr.size > 0 else 0
                summary[f"{key}_max"] = float(arr.max()) if arr.size > 0 else 0
                # 保存前100个元素用于精确对比
                flat = arr.flatten()[:100]
                summary[f"{key}_first100"] = flat.tolist()
        elif isinstance(value, (list, tuple)):
            summary[f"{key}_len"] = len(value)
            summary[f"{key}_content"] = str(value)[:500]  # 截断
        elif value is None:
            summary[f"{key}"] = None
        else:
            summary[f"{key}"] = str(value)[:500]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"Saved batch summary to {output_file}")
    return summary


def run_end2end():
    """在 end2end 环境下运行"""
    print("=" * 70)
    print("Running in end2end mode")
    print("=" * 70)
    
    # 添加路径
    sys.path.insert(0, os.getcwd())
    
    import torch
    from recovlm.data.dataloaders_v2 import get_dataloader as get_dataloader_v2
    
    # 加载配置
    config_path = "examples/vq_end2end_video/run_exp0.0.1_stage1.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print(f"Config: {json.dumps(config, indent=2)}")
    
    # 设置必要参数
    config['rank'] = 0
    config['world_size'] = 1
    
    # 创建数据集
    try:
        name = config.pop('name', 'chat_vision_parquet')
        dataloader = get_dataloader_v2(name=name, **config)
        
        # 获取一个 batch
        print("\nGetting first batch...")
        data_iter = iter(dataloader)
        batch = next(data_iter)
        
        print(f"\nBatch keys: {list(batch.keys())}")
        for key, value in batch.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            elif isinstance(value, (list, tuple)):
                print(f"  {key}: list/tuple, len={len(value)}")
            else:
                print(f"  {key}: {type(value)}")
        
        # 保存摘要
        output_file = "/llm_reco/maosiyang/temp/batch_summary_end2end.json"
        save_batch_summary(batch, output_file)
        
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()


def run_msy_master_2():
    """在 msy_master_2 环境下运行"""
    print("=" * 70)
    print("Running in msy_master_2 mode")
    print("=" * 70)
    
    # 添加路径
    sys.path.insert(0, os.getcwd())
    
    import torch
    from muse.data.datasets import ChatCompletionVisionDataset_keye_vitrope_slowfast_video
    from torch.utils.data import DataLoader
    
    # 加载配置
    config_path = "examples/keye_tokenizer_end2end_video/run_exp1.6.8_stage1.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print(f"Config: {json.dumps(config, indent=2)}")
    
    # 设置必要参数
    config['rank'] = 0
    config['world_size'] = 1
    
    # 创建数据集
    try:
        dataset = ChatCompletionVisionDataset_keye_vitrope_slowfast_video(**config)
        
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,  # 单线程便于调试
            collate_fn=lambda x: x[0]
        )
        
        # 获取一个 batch
        print("\nGetting first batch...")
        data_iter = iter(dataloader)
        batch = next(data_iter)
        
        print(f"\nBatch keys: {list(batch.keys())}")
        for key, value in batch.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            elif isinstance(value, (list, tuple)):
                print(f"  {key}: list/tuple, len={len(value)}")
            else:
                print(f"  {key}: {type(value)}")
        
        # 保存摘要
        output_file = "/llm_reco/maosiyang/temp/batch_summary_msy_master_2.json"
        save_batch_summary(batch, output_file)
        
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()


def compare_summaries():
    """对比两个摘要文件"""
    print("=" * 70)
    print("Comparing batch summaries")
    print("=" * 70)
    
    end2end_file = "/llm_reco/maosiyang/temp/batch_summary_end2end.json"
    msy_file = "/llm_reco/maosiyang/temp/batch_summary_msy_master_2.json"
    
    if not os.path.exists(end2end_file):
        print(f"Missing: {end2end_file}")
        print("Please run: cd end2end/muse && python ../../verify_dataset_real_output.py --mode end2end")
        return
    
    if not os.path.exists(msy_file):
        print(f"Missing: {msy_file}")
        print("Please run: cd msy_master_2/muse && python ../../verify_dataset_real_output.py --mode msy_master_2")
        return
    
    with open(end2end_file, 'r', encoding='utf-8') as f:
        end2end = json.load(f)
    
    with open(msy_file, 'r', encoding='utf-8') as f:
        msy = json.load(f)
    
    print("\n## Keys comparison:")
    end2end_keys = set(end2end.get('keys', []))
    msy_keys = set(msy.get('keys', []))
    
    common_keys = end2end_keys & msy_keys
    only_end2end = end2end_keys - msy_keys
    only_msy = msy_keys - end2end_keys
    
    print(f"  Common keys: {common_keys}")
    print(f"  Only in end2end: {only_end2end}")
    print(f"  Only in msy_master_2: {only_msy}")
    
    print("\n## Shape comparison for common keys:")
    for key in sorted(common_keys):
        shape_key = f"{key}_shape"
        if shape_key in end2end and shape_key in msy:
            e2e_shape = end2end[shape_key]
            msy_shape = msy[shape_key]
            match = "✅" if e2e_shape == msy_shape else "❌"
            print(f"  {key}: end2end={e2e_shape}, msy={msy_shape} {match}")
    
    print("\n## Value comparison for critical fields:")
    critical_fields = ['input_ids', 'loss_mask', 'position_ids', 'attention_mask']
    for key in critical_fields:
        if key in common_keys:
            first100_key = f"{key}_first100"
            if first100_key in end2end and first100_key in msy:
                e2e_vals = end2end[first100_key]
                msy_vals = msy[first100_key]
                
                # 对比前N个元素
                n = min(20, len(e2e_vals), len(msy_vals))
                match_count = sum(1 for i in range(n) if e2e_vals[i] == msy_vals[i])
                
                print(f"\n  {key}:")
                print(f"    end2end first 20: {e2e_vals[:20]}")
                print(f"    msy_master_2 first 20: {msy_vals[:20]}")
                print(f"    Match: {match_count}/{n}")
    
    print("\n## position_ids detailed comparison (critical for RoPE):")
    pos_key = "position_ids_first100"
    if pos_key in end2end and pos_key in msy:
        e2e_pos = np.array(end2end[pos_key])
        msy_pos = np.array(msy[pos_key])
        
        print(f"  end2end shape hint: {end2end.get('position_ids_shape', 'N/A')}")
        print(f"  msy_master_2 shape hint: {msy.get('position_ids_shape', 'N/A')}")
        
        if len(e2e_pos) == len(msy_pos):
            diff = np.abs(e2e_pos - msy_pos)
            print(f"  Max diff: {diff.max()}")
            print(f"  Mean diff: {diff.mean()}")
            if diff.max() > 0:
                print(f"  ❌ position_ids 有差异！这会导致 RoPE 计算不同！")
            else:
                print(f"  ✅ position_ids 一致")
        else:
            print(f"  ❌ 长度不同: {len(e2e_pos)} vs {len(msy_pos)}")


def main():
    parser = argparse.ArgumentParser(description='Verify dataset output equivalence')
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

