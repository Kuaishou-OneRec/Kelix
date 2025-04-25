import pyarrow
pyarrow.jemalloc_set_decay_ms(0)
import sys
import os
import uuid
import time
import traceback
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from utils import MPIBase
import gc
import psutil
import humanize
from multiprocessing import Process, Queue
from concurrent.futures import ThreadPoolExecutor
import concurrent

# pa.jemalloc_set_decay_ms(100)


# def convert_pandas_to_table(df):
#     t = pa.Table.from_pandas(df)
#     return t


def write_df_to_hdfs(df, file_path):
    from fastparquet import write
    tmp_fn = f"/code/{file_path.split('/')[-1]}"
    write(tmp_fn, df)
    os.system(f"/home/hadoop/software/hadoop/bin/hadoop fs -put {tmp_fn} {file_path}")
    os.remove(tmp_fn)


tmp_pd = None

def pq2pd_v2(x, rank=0):
    '''
    x =pq2pd_v2('viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage1/ASV2/rank-180-6ccefc5c-1cf8-11f0-a4bf-946daee90af8.parquet')
    '''
    import uuid
    import os
    import pyarrow.parquet as pq
    import time
    import subprocess
    from fastparquet import ParquetFile

    os.makedirs("/code/.pq_cache", exist_ok=True)
    tmp_fn = f"/code/.pq_cache/{uuid.uuid4()}_{rank}.parquet"
    for t in range(5):
        if x.startswith("viewfs://"):
            cmd = f"/home/hadoop/software/hadoop/bin/hadoop fs -get {x} {tmp_fn}"
            os.system(cmd)
            if not os.path.exists(tmp_fn):
                print(f"Retrying{rank} the {t} time to get {x} -> {tmp_fn}...")
                time.sleep(np.random.rand() * 10)
                continue
            df = ParquetFile(tmp_fn).to_pandas()
            break
        else:
            df = ParquetFile(x).to_pandas()
            break
    if not os.path.exists(tmp_fn):
        try:
            print("fall back to pq.read_table")
            return pq.read_table(x).to_pandas()
        except Exception as e:
            print(traceback.format_exc())
            raise FileNotFoundError(f"Failed to get {x} from HDFS. cmd={cmd}")
    if x.startswith("viewfs://"): os.remove(tmp_fn)
    return df



def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss  # 当前进程的 Resident Set Size (RSS)
    return humanize.naturalsize(mem, binary=True)  # 转换为易读格式（如GB/MB）


class AutoShuffler(MPIBase):
    def __init__(
        self,
        input_dir,
        output_dir,
        buffer_mem_size=2*1024,
        target_partition_size=30,
        shard_output_by_rank=True,
    ):
        super().__init__()
        # 输入输出配置
        self.raw_input_dir = input_dir
        self.final_output_dir = output_dir
        self.prepare_output_dir = os.path.join(output_dir, "_prepared")
        self.shard_output_by_rank = shard_output_by_rank
        
        # 内存管理
        self.buffer_mem_size = buffer_mem_size
        self.target_partition_size = target_partition_size
        
        # 文件系统连接
        try:
            self.fs = pa.hdfs.connect(user="mpi")
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            
        self.sample_rate_dict = {}
        
        # 自动创建目录结构
        if self.rank == 0:
            for d in [self.final_output_dir, self.prepare_output_dir]:
                self.mkdir(d)
                
        self.comm.barrier()
        # 调试：打印当前rank的初始化信息
        print(f"Rank-{self.rank} initialized. Memory usage: {get_memory_usage()}")

    def mkdir(self, *d, drop_last=False):
        # 调试：打印创建目录的rank和路径
        d = os.path.join(*d)
        print(f"Rank-{self.rank} creating directory: {d}")
        d0 = d
        if drop_last:
            d = os.path.dirname(d)
        if d.startswith("viewfs"):
            if not self.fs.exists(d): self.fs.mkdir(d)
        else:
            os.makedirs(d, exist_ok=True)
        return d0

    def ls(self, d):
        if d.startswith("viewfs://"):
            return self.fs.ls(d)
        else:
            return [os.path.join(d, x) for x in os.listdir(d)]
        
    def mv(self, src, dest):
        if src.startswith("viewfs://"):
            self.fs.mv(src, dest)
        else:
            os.rename(src, dest)

    def rm(self, path):
        try:
            print(f"going to rm {path}, length={len(self.ls(path))}")
        except Exception as e:
            print(traceback.format_exc())

        if path.startswith("viewfs://"):
            self.fs.rm(path, recursive=True)
        else:
            import shutil
            shutil.rmtree(path)
            
    def collect_parquet_files(self, input_dirs):
        """
        多线程收集Parquet文件列表
        :param input_dirs: 输入目录列表（支持@采样率语法）
        :return: 符合条件的Parquet文件列表
        """
        all_files = []
        sample_rate_dict = {}

        def process_directory(dir_path):
            try:
                original_dir = dir_path
                rate = 1.0
                if "@" in dir_path:
                    original_dir, rate_str = dir_path.split("@")
                    rate = float(rate_str)
                    sample_rate_dict[original_dir] = rate
                files = [f for f in self.ls(original_dir) if f.endswith(".parquet")]
                print(f"Processed {len(files)} files in {original_dir} with rate {rate}")
                return files
            except Exception as e:
                print(f"Error processing {dir_path}: {str(e)}")
                traceback.print_exc()
                return []

        # 创建进度条
        progress_bar = tqdm(
            total=len(input_dirs),
            desc="Collecting files",
            disable=not (self.rank == 0),
            leave=False
        )

        # 多线程执行目录处理
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(process_directory, d) for d in input_dirs]
            for future in futures:
                files = future.result()
                all_files.extend(files)
                progress_bar.update(1)

        progress_bar.close()
        self.sample_rate_dict = sample_rate_dict
        return all_files
    
    def _prepare_stage(self):
        """预处理阶段：生成均匀大小的中间文件"""
        import tqdm

        # 调试：阶段开始日志
        print(f"Rank-{self.rank} entering _prepare_stage. Memory: {get_memory_usage()}")

        # 收集文件列表（带采样率解析）
        if self.rank == 0:
            all_files = []
            all_files = self.collect_parquet_files(self.raw_input_dir)
            np.random.shuffle(all_files)
            print(f"Rank-0 prepared {len(all_files)} files. Memory: {get_memory_usage()}")
        else:
            all_files = None
        
        # 分发文件列表
        all_files = self.comm.bcast(all_files, root=0)
        my_files = all_files[self.rank::self.world_size]
        print(f"Rank-{self.rank} received {len(my_files)} files. Memory: {get_memory_usage()}")

        # 处理文件并生成均匀分块
        buffer = []
        
        for fpath in tqdm.tqdm(my_files, desc=f"Rank-{self.rank} Preprocessing"):
            try:
                # 内存监控：处理文件前
                print(f"Rank-{self.rank} processing {fpath}. Memory before: {get_memory_usage()}")
                

                # 读取并采样
                dirname = os.path.dirname(fpath)
                df = pq2pd_v2(fpath, self.rank)
                df = df.sample(frac=self.sample_rate_dict.get(dirname, 1.0))
                
                # 内存监控：读取文件后
                print(f"Rank-{self.rank} read {len(df)} rows from {fpath}. Memory after read: {get_memory_usage()}")
                
                # 分块写入
                for i in range(0, len(df), 128):  # 每128个样本一组
                    chunk = df.iloc[i:i+128]
                    buffer.append(chunk)
                    if len(buffer) * 128 >= self.buffer_mem_size:
                        self._flush_buffer(buffer, self.prepare_output_dir)
                        buffer = []
                        gc.collect()
                        print(f"Rank-{self.rank} flushed buffer-{i}. Memory after flush: {get_memory_usage()}")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Error processing {fpath}: {str(e)}")
        
        # 写入剩余数据
        if buffer:
            self._flush_buffer(buffer, self.prepare_output_dir)
            buffer = []
            gc.collect()
        print(f"Rank-{self.rank} finished _prepare_stage. Memory: {get_memory_usage()}")

    def _shuffle_stage(self):
        """混洗阶段：全局哈希混洗"""
        print(f"Rank-{self.rank} entering _shuffle_stage. Memory: {get_memory_usage()}")
        
        # 获取预处理后的文件列表
        if self.rank == 0:
            prepared_files = [
                f for f in self.ls(self.prepare_output_dir)
                if f.endswith(".parquet")
            ]
            np.random.shuffle(prepared_files)
            print(f"Rank-0 found {len(prepared_files)} prepared files. Memory: {get_memory_usage()}")
        else:
            prepared_files = None
        prepared_files = self.comm.bcast(prepared_files, root=0)
        my_files = prepared_files[self.rank::self.world_size]
        print(f"Rank-{self.rank} received {len(my_files)} shuffled files. Memory: {get_memory_usage()}")

        # 全局混洗处理
        global_buffer = []
        global_buffer_len = 0
        for fpath in tqdm(my_files, desc=f"Rank-{self.rank} Shuffling"):
            try:
                # 内存监控：处理文件前
                print(f"Rank-{self.rank} shuffling {fpath}. Memory before: {get_memory_usage()}")
                
                # df = pq.read_table(fpath).to_pandas()
                df = pq2pd_v2(fpath)
                # df["_shard_hash"] = df.uuid.apply(hash)
                global_buffer.append(df)
                global_buffer_len += len(df)
                
                # 内存监控：读取文件后
                print(f"Rank-{self.rank} added {len(df)} rows from {fpath}. Memory after read: {get_memory_usage()}")
                
                if global_buffer_len >= self.target_partition_size * 10:
                    merged = pd.concat(global_buffer , copy=False)
                    merged = merged.sample(frac=1.0) # frac=self.sample_rate_dict.get(dirname, 1.0))
                    # merged.sort_values("_shard_hash", inplace=True)
                    self._write_shuffled(merged)
                    global_buffer = []
                    global_buffer_len = 0
                    gc.collect()
                    print(f"Rank-{self.rank} flushed global buffer. Memory after flush: {get_memory_usage()}")
            except Exception as e:
                print(f"Error shuffling {fpath}: {str(e)}")
        
        # 处理剩余数据
        if global_buffer:
            merged = pd.concat(global_buffer, copy=False)
            merged = merged.sample(frac=1.0)
            # merged.sort_values("_shard_hash", inplace=True)
            self._write_shuffled(merged)
            global_buffer = []
            gc.collect()
        print(f"Rank-{self.rank} finished _shuffle_stage. Memory: {get_memory_usage()}")

    def _flush_buffer(self, buffer, output_dir):
        """写入缓冲数据"""
        if not buffer:
            return

        combined = pd.concat(buffer, copy=False)
        filename = f"prep-{self.rank}-{uuid.uuid4().hex}.parquet"
        # tmp_path = os.path.join(self.mkdir(output_dir, ".tmp"), filename)
        final_path = os.path.join(output_dir, str(self.rank), filename) if self.shard_output_by_rank else os.path.join(output_dir, filename)
        
        # 调试：写入缓冲日志
        # print(f"Rank-{self.rank} flushing {len(combined)} rows to {final_path}. Memory: {get_memory_usage()}")
        
        # 保证原子写入
        # t = convert_pandas_to_table(combined)
        # t = 0
        # print(type(t))
        # pq.write_table(t, tmp_path, row_group_size=2048)
        write_df_to_hdfs(combined, final_path)

        print(f"Rank-{self.rank} flushing {len(combined)} rows to {final_path} after. Memory: {get_memory_usage()}")
        # gc.collect()
        # print(f"Rank-{self.rank} after garbage collection. Memory: {get_memory_usage()}")

        time.sleep(0.5)  # 防止HDFS元数据延迟
        # self.mv(tmp_path, final_path)
        # pa.default_memory_pool().release_unused()
        # print(f"Rank-{self.rank} after release_unused. Memory: {get_memory_usage()}")

    def _write_shuffled(self, df):
        """写入混洗后的数据"""
        chunk_size = self.target_partition_size
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i+chunk_size]
            filename = f"shuffle-{uuid.uuid4().hex}.parquet"
            final_path = os.path.join(self.final_output_dir, filename)
            
            # 调试：写入混洗数据日志
            print(f"Rank-{self.rank} writing {i}-th chunk ({len(chunk)}/{len(df)}) to {final_path}. Memory: {get_memory_usage()}")
            
            # pq.write_table(convert_pandas_to_table(chunk), final_path)
            write_df_to_hdfs(chunk, str(self.rank), final_path) if self.shard_output_by_rank else write_df_to_hdfs(chunk, final_path)
            time.sleep(0.3)

    def _collect_output_files(self, directory):
        """收集指定目录下的所有Parquet文件"""
        print(f"_collect_output_files ...")
        if self.shard_output_by_rank:
            all_files = self.get_all_files(directory)
        else:
            all_files = self.ls(directory)
        
        all_files = [x for x in all_files if x.endswith(".parquet")]
        
        print("=" * 30)
        import json
        print(json.dumps(all_files, indent=4))
        dump_path = '/code/__all_shuffle_v2_all_files.json'
        with open(dump_path, 'w') as json_file: json.dump(all_files, json_file, indent=4)
        os.system(f"/home/hadoop/software/hadoop/bin/hadoop fs -put {dump_path} {directory}")
        print(f"all files are written successfully {dump_path} -> {directory}")
        print("=" * 30)

    def get_files_for_rank(self, rank, directory):
        rank_dir = os.path.join(directory, str(rank))
        return self.ls(rank_dir)
    # 在类的方法中使用以下代码
    def get_all_files(self, directory):
        all_files = []
        with concurrent.futures.ThreadPoolExecutor(64) as executor:
            futures = []
            for rank in tqdm(range(self.world_size)):
                future = executor.submit(self.get_files_for_rank, rank, directory)
                futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                try:
                    files = future.result()
                    all_files += files
                except Exception as e:
                    print(f"Error getting files for rank: {e}")
        return all_files

    def run(self, stages=[1,2]):
        if 1 in stages:
            # 阶段1：预处理
            print(f"Rank-{self.rank} starting run. Memory: {get_memory_usage()}")
            self._prepare_stage()
            self.comm.barrier()  # 确保所有节点完成预处理
            
        if self.rank == 0:
            all_files = self._collect_output_files(self.prepare_output_dir)

        all_files = self.comm.bcast(all_files, root=0)
        if 2 in stages:
            # 阶段2：自动触发混洗
            self._shuffle_stage()
            self.comm.barrier()
        
        if self.rank == 0:
            all_files = self._collect_output_files(self.final_output_dir)

        # if self.rank == 0 and 0:
        #     self.rm(self.prepare_output_dir)
        #     all_files = self.ls(self.final_output_dir)
        #     print(f"Rank-0 completed. Total files: {len(all_files)}. Memory: {get_memory_usage()}")
        #     dump_path = '/code/AutoShuffler_v2/run_tmp.json'
        #     print(f"Rank-0 dumping {len(all_files)} files to {dump_path}. Memory: {get_memory_usage()}")
        #     import json
        #     with open(self.mkdir(dump_path, drop_last=True), 'w') as json_file:
        #         json.dump(all_files, json_file, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Auto Two-Stage Shuffler")
    parser.add_argument("--input", required=True, nargs="+", 
                       help="输入目录，支持@采样率语法（如 /data@0.5）")
    parser.add_argument("--output", required=True, 
                       help="最终输出目录")
    parser.add_argument("--buffer", type=int, default=512,
                       help="内存缓冲区大小（默认8GB）")
    parser.add_argument("--partition", type=int, default=2048,
                       help="目标分块大小（行数）")
    args = parser.parse_args()

    shuffler = AutoShuffler(
        input_dir=args.input,
        output_dir=args.output,
        buffer_mem_size=args.buffer,
        target_partition_size=args.partition
    )
    print(f"Main process initialized for Rank-{shuffler.rank}. Memory: {get_memory_usage()}")
    shuffler.run([1,2])


if __name__ == "__main__":
    main()


