#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import random
from recovlm.data.prompts import PromptLoader
import pyarrow as pa
import pyarrow.parquet as pq
from typing import List, Dict
import math

def create_sample(pid_info, prompt_loader, prompt_name=None):
    """为单个PID创建数据样本"""
    media_path = pid_info.get('media_path')
    if not media_path or not os.path.exists(media_path):
        return None
    
    # 使用PromptLoader加载prompt
    prompt = prompt_loader.load(prompt_name)
    if not prompt:
        prompt = "Describe this video."
    
    # 构建messages字段
    messages_data = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": f"file://{media_path}",
                    "max_pixels": 1280 * 28 * 28,
                    "min_pixels": 16 * 28 * 28,
                    "fps": 1.0
                },
                {"type": "text", "text": prompt}
            ]
        },
        {"role": "assistant", "content": ""}
    ]
        
    return {
        "images": json.dumps({}),  # empty map for image key -> base64
        "videos": json.dumps([media_path]),  # list of video paths
        "source": "creator_bot_comment_eval",
        "messages": json.dumps(messages_data),  # chat格式数据
        "segments": json.dumps([]),  # empty list for pretrain格式数据
        "metadata": json.dumps({}),  # empty map for meta信息
        "uuid": str(pid_info['pid'])  # 使用pid作为uuid
    }

def save_to_parquet(samples: List[Dict], output_path: str, num_shards: int = 1):
    """将样本保存为parquet格式，支持分片"""
    if not samples:
        print("No samples to save")
        return

    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # 计算每个分片的大小
    shard_size = math.ceil(len(samples) / num_shards)
    
    # 根据格式约定准备schema
    schema = pa.schema([
        ('images', pa.string()),     # json string, map<string, string>
        ('videos', pa.string()),     # json string, list of video paths
        ('source', pa.string()),     # 数据来源
        ('messages', pa.string()),   # json string, chat格式数据
        ('segments', pa.string()),   # json string, pretrain格式数据
        ('metadata', pa.string()),   # json string, map<string, string>
        ('uuid', pa.string())        # 样本uuid
    ])

    # 按分片保存数据
    for shard_id in range(num_shards):
        start_idx = shard_id * shard_size
        end_idx = min((shard_id + 1) * shard_size, len(samples))
        shard_samples = samples[start_idx:end_idx]

        # 构建分片文件名
        if num_shards > 1:
            shard_path = f"{output_path}.{shard_id:05d}-of-{num_shards:05d}.parquet"
        else:
            shard_path = f"{output_path}.parquet"

        # 转换数据为Arrow表格式
        table = pa.Table.from_pylist(shard_samples, schema=schema)
        
        # 写入parquet文件
        pq.write_table(table, shard_path)
        print(f"Saved shard {shard_id + 1}/{num_shards} with {len(shard_samples)} samples to {shard_path}")

def prepare_dataset(input_dir: str, output_path: str, prompt_name: str = None, num_shards: int = 1):
    """准备数据集并保存为parquet格式"""
    input_dir = Path(input_dir)

    # 初始化PromptLoader
    prompt_loader = PromptLoader()

    samples = []
    for json_file in input_dir.glob("*.json"):
        with open(json_file) as f:
            pid_info = json.load(f)
            sample = create_sample(pid_info, prompt_loader, prompt_name)
            if sample:
                samples.append(sample)

    print(f"Created {len(samples)} samples")
    save_to_parquet(samples, output_path, num_shards)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset from PID information")
    parser.add_argument("--input-dir", default="./output", 
                       help="Directory containing PID JSON files")
    parser.add_argument("--output-path", default="./output/dataset/dataset",
                       help="Output path for parquet files (without extension)")
    parser.add_argument("--prompt-name", default=None,
                       help="Name of the prompt file to use (without .txt extension)")
    parser.add_argument("--num-shards", type=int, default=1,
                       help="Number of shards to split the dataset into")
    
    args = parser.parse_args()
    prepare_dataset(args.input_dir, args.output_path, args.prompt_name, args.num_shards) 