#!/usr/bin/env python3
import pandas as pd
import json
import argparse
from pathlib import Path

def convert_xlsx_to_jsonl(input_file: str, output_file: str):
    """将xlsx文件转换为jsonl格式"""
    # 读取xlsx文件
    print(f"Reading xlsx file: {input_file}")
    df = pd.read_excel(input_file)
    
    # 确保输出目录存在
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 将每行转换为json并写入文件
    print(f"Converting to jsonl and saving to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            # 将所有字段转换为字符串格式，处理nan值
            data = {
                'author_id': str(row['author_id']) if pd.notna(row['author_id']) else '',
                'photo_id': str(row['photo_id']) if pd.notna(row['photo_id']) else '',
                'comment_id': str(row['comment_id']) if pd.notna(row['comment_id']) else '',
                'comment_info': str(row['comment_info']) if pd.notna(row['comment_info']) else '',
                'reply_id': str(row['reply_id']) if pd.notna(row['reply_id']) else '',
                'reply_info': str(row['reply_info']) if pd.notna(row['reply_info']) else '',
                'bot_id': str(row['bot_id']) if pd.notna(row['bot_id']) else ''
            }
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    print(f"Successfully converted {len(df)} rows")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert xlsx file to jsonl format")
    parser.add_argument("input_file", help="Input xlsx file path")
    parser.add_argument("--output-file", default="output.jsonl",
                       help="Output jsonl file path")
    
    args = parser.parse_args()
    convert_xlsx_to_jsonl(args.input_file, args.output_file) 