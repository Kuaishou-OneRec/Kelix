#!/usr/bin/env python3
import argparse
from recovlm.services.clients import PidInfoClient
import json
from pathlib import Path
from tqdm import tqdm
from recovlm.utils.media import get_pid_folder

def download_pid_info(pid_list_file: str, output_dir: str):
    """下载PID信息并保存为JSON文件"""
    client = PidInfoClient()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(pid_list_file, 'r') as f:
        pids = [line.strip() for line in f if line.strip()]

    for pid in tqdm(pids):
        output_file = get_pid_folder(pid, output_dir) / f"{pid}.json"
        #output_file = output_dir / f"{pid}.json"
        if output_file.exists():
            print(f"Skipping PID {pid} because it already exists")
            continue
            
        try:
            info = client.get_pid_info(int(pid))
            with open(output_file, 'w') as f:
                json.dump(info, f, indent=2)
            print(f"Downloaded info for PID: {pid}")
        except Exception as e:
            print(f"Error downloading PID {pid}: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download PID information")
    parser.add_argument("pid_list_file", help="File containing list of PIDs")
    parser.add_argument("--output-dir", default="./output", help="Output directory for JSON files")
    
    args = parser.parse_args()
    download_pid_info(args.pid_list_file, args.output_dir) 