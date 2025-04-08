#!/usr/bin/env python3
import json
import argparse
from pathlib import Path

def extract_photo_ids(input_file: str, output_file: str):
    """从jsonl文件中提取photo_id并保存为txt文件"""
    print(f"Reading jsonl file: {input_file}")
    
    # 确保输出目录存在
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 用set来去重
    photo_ids = set()
    
    # 读取jsonl文件并提取photo_id
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                photo_id = data.get('photo_id', '').strip()
                if photo_id:  # 只保存非空的photo_id
                    photo_ids.add(photo_id)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line: {e}")
                continue
    
    # 将photo_id写入txt文件
    print(f"Writing {len(photo_ids)} unique photo_ids to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for photo_id in sorted(photo_ids):  # 排序以保持稳定的输出顺序
            f.write(f"{photo_id}\n")
    
    print(f"Successfully extracted {len(photo_ids)} unique photo_ids")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract photo_ids from jsonl file")
    parser.add_argument("input_file", help="Input jsonl file path")
    parser.add_argument("--output-file", default="photo_ids.txt",
                       help="Output txt file path")
    
    args = parser.parse_args()
    extract_photo_ids(args.input_file, args.output_file) 