import json
import subprocess

def list_hdfs_files(hdfs_dir):
    """
    列出HDFS目录下的所有文件的完整路径，不处理包含 'test' 的parquet文件
    Args:
        hdfs_dir: HDFS目录路径
    Returns:
        list: 包含所有文件完整HDFS路径的列表
    """
    # 使用hadoop fs -ls命令列出HDFS目录下的文件
    result = subprocess.run(['hadoop', 'fs', '-ls', hdfs_dir], capture_output=True, text=True)
    files = []
    
    # 解析命令输出，提取完整文件路径
    for line in result.stdout.splitlines():
        if not line.strip() or 'Found' in line:  # 跳过空行和Found开头的行
            continue
        parts = line.split()
        if len(parts) >= 8:  # HDFS ls输出格式检查
            file_path = parts[-1]  # 获取完整路径
            if file_path == hdfs_dir:  # 排除目录本身
                continue
            # 如果文件名包含'test'且以 .parquet 结尾，则跳过该文件处理（忽略大小写）
            if file_path.lower().endswith('.parquet') and 'test' in file_path.lower():
                continue
            if file_path.lower().endswith('.parquet') :
                files.append(file_path)
    
    return files

def write_to_json(file_list, json_file):
    """
    将文件路径列表写入JSON文件（单行格式）
    Args:
        file_list: 文件路径列表
        json_file: 输出的JSON文件路径
    """
    with open(json_file, 'w') as f:
        json.dump(file_list, f, separators=(',', ':'))

def main():
    hdfs_dir = 'viewfs://hadoop-lt-cluster/home/reco_wl/mpi/huqigen/recovlm_dataset/wenjuan_sft/0210_11w_with_cot_dataset_only_match_llm_label_full_cmt_v2'  # 替换为你的HDFS目录路径
    json_file = '/llm_reco_ssd/huqigen/dataset/wenjuan_sft/photo_0210_11w/recovlm_dataset_wenjuan_0207_1k_with_cot_only_match_llm_label.json'  # 输出的JSON文件名
    
    file_list = list_hdfs_files(hdfs_dir)
    write_to_json(file_list, json_file)
    print(f"File paths have been written to {json_file}")
    print(f"Total files: {len(file_list)}")

if __name__ == '__main__':
    main()