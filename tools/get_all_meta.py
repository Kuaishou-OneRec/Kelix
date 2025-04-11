import sys
import subprocess
import json
from mpi4py import MPI
import pyarrow.parquet as pq
import pandas as pd

def get_meta_column_from_file(fn):
    try:
        # 读取parquet文件并只选择meta列
        parquet_file = pq.ParquetFile(fn)
        data = parquet_file.read(columns=['metadata']).to_pandas()
        return data['metadata'].tolist()
    except Exception as e:
        print(f"读取文件 {fn} 时出错: {e}")
        return []

def shell_hdfs_ls(source_dir):
  try:
    command = f"hdfs dfs -ls {source_dir}"
    result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
    files = []
    for line in result.stdout.splitlines():
      parts = line.split()
      if len(parts) > 0 and parts[-1].startswith('viewfs://'):
        files.append(parts[-1])
    return files

  except subprocess.CalledProcessError as e:
    # print(f"Error occurred: {traceback.format_exc()}")
    return []

def collect_meta_from_hdfs_folder(data_folder, local_meta_data):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
  
    if rank == 0:
        fn_list = shell_hdfs_ls(data_folder)
        all_files = [fn for fn in fn_list if fn.endswith(".parquet")]
    else:
        all_files = None

    all_files = comm.bcast(all_files, root=0)
    num_files = len(all_files)

    files_per_process = num_files // size
    remainder = num_files % size
    start_index = rank * files_per_process + min(rank, remainder)
    end_index = start_index + files_per_process + (1 if rank < remainder else 0)
    local_files = all_files[start_index:end_index]

    for file_path in local_files:
        file_meta = get_meta_column_from_file(file_path)
        local_meta_data.extend(file_meta)
    
    print(f'进程 {rank}: 从 {data_folder} 读取了 {len(local_meta_data)} 条meta数据')

if __name__ == '__main__':
    data_file = sys.argv[1]
    output_file_base =  "/llm_reco/chuchenglong/R3/asr_meta"
    
    hdfs_dirs = []
    with open(data_file) as fp:
        for line in fp:
            if line.strip() != "":
                hdfs_dirs.append(line.strip())
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # 用于收集当前进程的meta数据的列表
    local_meta_data = []
    
    for fn in hdfs_dirs:
        collect_meta_from_hdfs_folder(fn, local_meta_data)
    
    # 每个进程保存自己的数据
    output_file = f"{output_file_base}_rank{rank}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(local_meta_data, f, ensure_ascii=False, indent=2)
    print(f'进程 {rank}: 成功将 {len(local_meta_data)} 条meta数据保存到 {output_file}')