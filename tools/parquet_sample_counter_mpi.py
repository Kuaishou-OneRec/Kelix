import sys
import subprocess
from mpi4py import MPI
import pyarrow.parquet as pq

def count_rows_in_file(fn):
    parquet_file = pq.ParquetFile(fn)
    return parquet_file.metadata.num_rows
    # data = parquet_file.read().to_pandas()
    # return len(data)

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

def count_hdfs_folder(data_folder):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
  
    if rank == 0:
        fn_list = shell_hdfs_ls(data_folder)
        all_files = [fn for fn in fn_list if fn.endswith(".parquet")]
    else:
        all_files = None
        num_files = 0

    all_files = comm.bcast(all_files, root=0)
    num_files = len(all_files)

    files_per_process = num_files // size
    remainder = num_files % size
    start_index = rank * files_per_process + min(rank, remainder)
    end_index = start_index + files_per_process + (1 if rank < remainder else 0)
    local_files = all_files[start_index:end_index]

    local_row_count = 0
    for file_path in local_files:
        local_row_count += count_rows_in_file(file_path)
    total_row_count = comm.reduce(local_row_count, op=MPI.SUM, root=0)

    if rank == 0:
        return data_folder, total_row_count
    return None, None

if __name__ == '__main__':
    data_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "results.txt"
    
    hdfs_dirs = []
    with open(data_file) as fp:
        for line in fp:
            if line.strip() != "":
                hdfs_dirs.append(line.strip())
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    results = []
    for fn in hdfs_dirs:
        folder, count = count_hdfs_folder(fn)
        if rank == 0 and folder is not None and count is not None:
            results.append(f'{folder}\t{count}')
    
    if rank == 0:
        with open(output_file, 'w') as f:
            for result in results:
                f.write(result + '\n')