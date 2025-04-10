#!/usr/bin/env python3
import json
import pandas as pd
import argparse
from pathlib import Path

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def main():
    parser = argparse.ArgumentParser(description='Convert JSONL file to Excel')
    parser.add_argument('--input_file', type=str, required=True,
                        help='Input JSONL file path')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output Excel file path')
    parser.add_argument('--sheet_name', type=str, default='Sheet1',
                        help='Excel sheet name')
    args = parser.parse_args()

    # Load JSONL data
    data = load_jsonl(args.input_file)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Write to Excel
    df.to_excel(args.output_file, sheet_name=args.sheet_name, index=False, engine='openpyxl')
    
    print(f"Data has been written to {args.output_file}")

if __name__ == '__main__':
    main() 