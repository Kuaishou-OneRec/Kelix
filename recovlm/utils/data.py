from typing import List, Dict, Tuple
import os
import os.path as osp
import json
import pyarrow as pa
import pyarrow.parquet as pq
import math
from recovlm.utils.media import get_pid_folder, encode_image
from pathlib import Path

DEFAULT_CACHE_DIR = "/llm_reco/zhouyang12/.cache"
DEFAULT_PHOTO_DIR = DEFAULT_CACHE_DIR + "/Photo"

def get_media_info(pid: str, photo_dir: str = DEFAULT_PHOTO_DIR) -> Dict:
    """从json文件中获取媒体信息"""
    json_path = get_pid_folder(pid, Path(photo_dir)) / f"{pid}.json"
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
    """创建MP4视频类型的video content"""
    return [
        {
            "type": "video",
            "video": media_path
        }
    ]

def create_images_content(image_paths: List[str]) -> List[Dict]:
    """创建多图类型的video content"""
    return [
        {
            "type": "video",
            "video": [
                f"{i}.jpg" for i in range(len(image_paths))
            ]
        }
    ]

def create_media_content(media_path: str) -> Tuple[List[Dict], str, str]:
    is_mp4 = isinstance(media_path, str) and media_path.endswith('.mp4')

    if is_mp4:
        if not os.path.exists(media_path):
            print(f"Warning: Video file not found: {media_path}")
            return None
        if not osp.exists(media_path) or osp.getsize(media_path) == 0:
            print(f"Warning: Video file is empty: {media_path}")
            return None
        content = create_video_content(media_path)
        videos_json = json.dumps({})
        images_json = json.dumps({})
    else:
        valid_paths = [path for path in media_path if os.path.exists(path)]
        if not valid_paths:
            print(f"Warning: No valid image files found for PID {pid}")
            return None

        content = create_images_content(valid_paths)
        # 多图需要编码成base64
        images_json = json.dumps({
            f"{i}.jpg": encode_image(img) for i, img in enumerate(valid_paths)
        })
        videos_json = json.dumps({})
    
    return content, videos_json, images_json

def save_to_parquet(samples: List[Dict], output_dir: str, num_shards: int = 1):
    """将样本保存为parquet格式，支持分片"""
    if not samples:
        print("No samples to save")
        return

    # 确保输出目录存在
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
        shard_path = f"{output_dir}/part-{shard_id:05d}-of-{num_shards:05d}.parquet"
        generated_files.append(shard_path)
        # 转换数据为Arrow表格式
        table = pa.Table.from_pylist(shard_samples, schema=schema)
        
        # 写入parquet文件
        pq.write_table(table, shard_path)
        print(
            f"Saved shard {shard_id + 1}/{num_shards} with "
            f"{len(shard_samples)} samples to {shard_path}")
    
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

def create_dataset_config(index_path: str,
                          output_dir: str,
                          tokenizer_path: str = None,
                          name: str = "vllm_infer",
                          min_visual_tokens_per_image: int = 4,
                          max_visual_tokens_per_image: int = 1024,
                          video_fps: float = 1.0,
                          video_min_frames: int = 2,
                          video_max_frames: int = 60,
                          max_images: int = 30,
                          num_workers: int = 4):
    """创建dataset_config文件"""
    config = {
        "name": name,
        "sources": index_path,
        "min_visual_tokens_per_image": min_visual_tokens_per_image,
        "max_visual_tokens_per_image": max_visual_tokens_per_image,
        "video_fps": video_fps,
        "video_min_frames": video_min_frames,
        "video_max_frames": video_max_frames,
        # TODO: support more models
        "pretrained_model_name_or_path": tokenizer_path,
        "max_images": max_images,
        "num_workers": num_workers,
        "shrink_ratio": 0.7
    }
    
    # 保存在与parquet文件相同的目录下
    config_path = os.path.join(output_dir, "dataset_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Created dataset config at {config_path}")
    return config_path

def save_parquet_dataset(samples: List[Dict], output_dir: str, num_shards: int,
                         tokenizer_path: str = None,
                         name: str = "vllm_infer",
                         min_visual_tokens_per_image: int = 4,
                         max_visual_tokens_per_image: int = 1024,
                         video_fps: float = 1.0,
                         video_min_frames: int = 2,
                         video_max_frames: int = 60,
                         max_images: int = 30,
                         num_workers: int = 4):
        # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    parquet_files = save_to_parquet(samples, output_dir, num_shards)
    
    if parquet_files:
        # 获取parquet文件所在的目录（应该与output_path相同）
        parquet_dir = os.path.dirname(parquet_files[0])
        
        # 创建索引文件，并保存在parquet目录下
        index_path = create_index_file(parquet_files, parquet_dir)
        
        # 创建dataset_config文件，并保存在parquet目录下
        create_dataset_config(index_path, parquet_dir, tokenizer_path,
                              name,
                              min_visual_tokens_per_image,
                              max_visual_tokens_per_image,
                              video_fps,
                              video_min_frames,
                              video_max_frames,
                              max_images, num_workers)
    else:
        print("No parquet files were generated")