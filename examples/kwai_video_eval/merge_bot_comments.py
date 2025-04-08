#!/usr/bin/env python3
import json
import glob
import os
import argparse
from pathlib import Path

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def main():
    parser = argparse.ArgumentParser(description='Merge bot comments with rank files')
    parser.add_argument('--bot_comment_file', type=str, default=None,
                        help='Path of the bot_comment.jsonl file')
    parser.add_argument('--response_dir', type=str, 
                        default=None,
                        help='Path of the response directory')
    parser.add_argument('--output_file', type=str, default='merged_bot_comments.jsonl',
                        help='Output file path')
    args = parser.parse_args()

    # Load bot_comment.jsonl
    bot_comments = load_jsonl(args.bot_comment_file)
    
    # Load all rank_*.jsonl files
    rank_files = glob.glob(os.path.join(args.response_dir, 'rank_*'))
    
    # Create a dictionary to store responses by photo_id
    responses_by_photo = {}
    for rank_file in rank_files:
        rank_data = load_jsonl(rank_file)
        for item in rank_data:
            photo_id = item['__key__']
            responses_by_photo[photo_id] = item['responses']
    
    # Merge the data
    merged_data = []
    for comment in bot_comments:
        photo_id = comment['photo_id']
        if str(photo_id) in responses_by_photo:
            comment['responses'] = responses_by_photo[str(photo_id)]
        merged_data.append(comment)
    
    # Write the merged data to a new file
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for item in merged_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Merged data has been written to {args.output_file}")

if __name__ == '__main__':
    main() 