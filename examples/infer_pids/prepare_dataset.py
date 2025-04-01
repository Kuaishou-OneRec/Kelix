#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

def create_sample(pid_info):
    """为单个PID创建数据样本"""
    media_path = pid_info.get('media_path')
    if not media_path or not os.path.exists(media_path):
        return None
        
    return {
        "__key__": str(pid_info['pid']).zfill(9),
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": f"file://{media_path}",
                        "max_pixels": 360 * 420,
                        "fps": 1.0,
                        "video_start": 0,
                        "video_end": None
                    },
                    {"type": "text", "text": "Describe this video."}
                ]
            },
            {"role": "assistant", "content": ""}
        ],
        "source": "kwai_video"
    }

def prepare_dataset(input_dir: str, output_file: str):
    """准备数据集并保存为JSONL格式"""
    input_dir = Path(input_dir)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    samples = []
    for json_file in input_dir.glob("*.json"):
        with open(json_file) as f:
            pid_info = json.load(f)
            sample = create_sample(pid_info)
            if sample:
                samples.append(sample)

    with open(output_file, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')

    print(f"Created dataset with {len(samples)} samples")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset from PID information")
    parser.add_argument("--input-dir", default="./output", help="Directory containing PID JSON files")
    parser.add_argument("--output-file", default="./output/dataset/dataset.jsonl", 
                       help="Output JSONL file path")
    
    args = parser.parse_args()
    prepare_dataset(args.input_dir, args.output_file) 