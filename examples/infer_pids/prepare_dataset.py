#!/usr/bin/env python3
import json
import os
from pathlib import Path
import argparse
from recovlm.data.prompts import PromptLoader
import pyarrow as pa
import pyarrow.parquet as pq
from typing import List, Dict
import math
import glob

def get_media_info(pid: str, photo_dir: str) -> Dict:
    """从json文件中获取媒体信息"""
    json_path = os.path.join(photo_dir, f"{pid}.json")
    if not os.path.exists(json_path):
        return None
    
    try:
        with open(json_path, 'r') as f:
            info = json.load(f)
            
        # 检查必要字段
        if not info.get('success') or info.get('error') is not None:
            print(f"Warning: Invalid info for PID {pid}: {info.get('error')}")
            return None
            
        if not info.get('media_path'):
            print(f"Warning: No media_path found for PID {pid}")
            return None
            
        return info
    except Exception as e:
        print(f"Error reading json for PID {pid}: {str(e)}")
        return None

def create_video_content(media_path: str) -> List[Dict]:
    """创建视频类型的content"""
    return [
        {
            "type": "video",
            "video": media_path",
            "max_pixels": 1280 * 28 * 28,
            "min_pixels": 16 * 28 * 28,
            "fps": 1.0
        }
    ]

def create_images_content(image_paths: List[str]) -> List[Dict]:
    """创建多图类型的content"""
    return [
        {
            "type": "image",
            "image": path,  # 直接使用图片的绝对路径
            "max_pixels": 1280 * 28 * 28,
            "min_pixels": 16 * 28 * 28,
        } for path in image_paths
    ]

def create_sample(pid: str, info: Dict, prompt_loader, prompt_name=None):
    """为单个PID创建数据样本"""
    media_path = info['media_path']
    if not media_path:
        print(f"Warning: Empty media_path for PID {pid}")
        return None

    # 使用PromptLoader加载prompt
    prompt = prompt_loader.load(prompt_name)
    if not prompt:
        prompt = "Describe this content."

    # 判断媒体类型
    is_video = isinstance(media_path, str) and media_path.endswith('.mp4')

    if is_video:
        if not os.path.exists(media_path):
            print(f"Warning: Video file not found: {media_path}")
            return None
        content = create_video_content(media_path)
        videos_json = json.dumps({})
        images_json = json.dumps({})
    else:
        # 检查所有图片是否存在
        valid_paths = [path for path in media_path if os.path.exists(path)]
        if not valid_paths:
            print(f"Warning: No valid image files found for PID {pid}")
            return None

        content = create_images_content(valid_paths)
        # 直接使用图片的绝对路径列表
        images_json = json.dumps({})
        videos_json = json.dumps({})

    # 构建messages
    messages_data = [
        {
            "role": "user",
            "content": content + [{"type": "text", "text": prompt}]
        },
        {"role": "assistant", "content": ""}
    ]

    return {
        "images": images_json,
        "videos": videos_json,
        "source": "kwai_video",
        "messages": json.dumps(messages_data),
        "segments": json.dumps([]),
        "metadata": json.dumps({}),
        "uuid": pid
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
    
    # 准备schema
    schema = pa.schema([
        ('images', pa.string()),     # json string, map<string, string>
        ('videos', pa.string()),     # json string, list of video paths
        ('source', pa.string()),     # 数据来源
        ('messages', pa.string()),   # json string, chat格式数据
        ('segments', pa.string()),   # json string, pretrain格式数据
        ('metadata', pa.string()),   # json string, map<string, string>
        ('uuid', pa.string())        # 样本uuid
    ])

    # 存储所有生成的parquet文件路径
    generated_files = []

    # 按分片保存数据
    for shard_id in range(num_shards):
        start_idx = shard_id * shard_size
        end_idx = min((shard_id + 1) * shard_size, len(samples))
        shard_samples = samples[start_idx:end_idx]

        # 构建分片文件名 (使用part-前缀)
        shard_path = f"{output_path}/part-{shard_id:05d}-of-{num_shards:05d}.parquet"
        generated_files.append(shard_path)

        # 转换数据为Arrow表格式
        table = pa.Table.from_pylist(shard_samples, schema=schema)
        
        # 写入parquet文件
        pq.write_table(table, shard_path)
        print(f"Saved shard {shard_id + 1}/{num_shards} with {len(shard_samples)} samples to {shard_path}")
    
    return generated_files

def create_index_file(parquet_files: List[str], output_dir: str):
    """创建索引文件，包含所有parquet文件的路径"""
    # 使用绝对路径
    absolute_paths = [os.path.abspath(f) for f in parquet_files]
    
    # 创建索引文件，保存在与parquet文件相同的目录下
    index_path = os.path.join(output_dir, "index.json")
    with open(index_path, 'w') as f:
        json.dump(absolute_paths, f, indent=2)
    
    print(f"Created index file at {index_path}")
    return index_path

def create_dataset_config(index_path: str, output_dir: str, model_path: str = None, name: str = "vllm_infer"):
    """创建dataset_config文件"""
    config = {
        "name": name,
        "sources": index_path,
        "min_visual_tokens_per_image": 4,
        "max_visual_tokens_per_image": 1024,
        "pretrained_model_name_or_path": model_path or "/llm_reco_ssd/zhouyang12/models/Qwen2-VL-7B-Instruct",
        "max_images": 10,
        "num_workers": 4
    }
    
    # 保存在与parquet文件相同的目录下
    config_path = os.path.join(output_dir, "dataset_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Created dataset config at {config_path}")
    return config_path

def prepare_dataset(pid_list_file: str, output_path: str,
                    photo_dir: str,
                    prompt_name: str = None,
                    num_shards: int = 1,
                    model_path: str = None):
    """从PID列表准备数据集并保存为parquet格式"""
    # 初始化PromptLoader
    prompt_loader = PromptLoader()

    # 读取PID列表
    with open(pid_list_file, 'r') as f:
        pids = [line.strip() for line in f if line.strip()]

    print(f"Found {len(pids)} PIDs in list")

    # 创建样本
    samples = []
    for pid in pids:
        try:
            info = get_media_info(pid, photo_dir)
            if info:
                sample = create_sample(pid, info, prompt_loader, prompt_name)
                if sample:
                    samples.append(sample)
            else:
                print(f"Warning: No valid info found for PID {pid}")
        except Exception as e:
            print(f"Error processing PID {pid}: {str(e)}")
            continue

    print(f"Created {len(samples)} valid samples")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    if samples:
        # 保存parquet文件并获取生成的文件列表
        parquet_files = save_to_parquet(samples, output_path, num_shards)
        
        if parquet_files:
            # 获取parquet文件所在的目录（应该与output_path相同）
            parquet_dir = os.path.dirname(parquet_files[0])
            
            # 创建索引文件，并保存在parquet目录下
            index_path = create_index_file(parquet_files, parquet_dir)
            
            # 创建dataset_config文件，并保存在parquet目录下
            create_dataset_config(index_path, parquet_dir, model_path)
        else:
            print("No parquet files were generated")
    else:
        print("No valid samples to save")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset from PID list")
    parser.add_argument("pid_list_file", 
                       help="File containing list of PIDs")
    parser.add_argument("--output-path", default="./output/dataset/dataset",
                       help="Output path for parquet files (without extension)")
    parser.add_argument("--photo-dir", default="/llm_reco/zhouyang12/.cache/Photo",
                       help="Cache directory containing Photo folder")
    parser.add_argument("--prompt-name", default=None,
                       help="Name of the prompt to use")
    parser.add_argument("--num-shards", type=int, default=1,
                       help="Number of shards to split the dataset into")
    parser.add_argument("--model-path", default=None,
                       help="Path to the model for dataset config")
    
    args = parser.parse_args()
    prepare_dataset(
        args.pid_list_file, args.output_path, args.photo_dir, 
        args.prompt_name, args.num_shards, args.model_path)