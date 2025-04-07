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

pa.jemalloc_set_decay_ms(0)

class AutoShuffler(MPIBase):
    def __init__(
        self,
        input_dir,
        output_dir,
        buffer_mem_size=2*1024,
        target_partition_size=2048,
    ):
        super().__init__()
        # 输入输出配置
        self.raw_input_dir = input_dir
        self.final_output_dir = output_dir
        self.prepare_output_dir = os.path.join(output_dir, "_prepared")
        
        # 内存管理
        self.buffer_mem_size = buffer_mem_size
        self.target_partition_size = target_partition_size
        
        # 文件系统连接
        self.fs = pa.hdfs.connect(user="mpi")
        self.sample_rate_dict = {}
        
        # 自动创建目录结构
        if self.rank == 0:
            for d in [self.final_output_dir, self.prepare_output_dir]:
                if not self.fs.exists(d):
                    self.fs.mkdir(d)
        self.comm.barrier()

    def _prepare_stage(self):
        """预处理阶段：生成均匀大小的中间文件"""
        # 收集文件列表（带采样率解析）
        if self.rank == 0:
            all_files = []
            for d in self.raw_input_dir:
                original_dir = d
                if "@" in d:
                    dir_path, rate = d.split("@")
                    self.sample_rate_dict[dir_path] = float(rate)
                    original_dir = dir_path
                all_files.extend([
                    f for f in self.fs.ls(original_dir)
                    if f.endswith(".parquet")
                ])
            np.random.shuffle(all_files)
        else:
            all_files = None
        
        # 分发文件列表
        all_files = self.comm.bcast(all_files, root=0)
        my_files = all_files[self.rank::self.world_size]

        # 处理文件并生成均匀分块
        buffer = []
        for fpath in tqdm(my_files, desc=f"Rank-{self.rank} Preprocessing"):
            try:
                # 读取并采样
                dirname = os.path.dirname(fpath)
                df = pq.read_table(fpath).to_pandas()
                df = df.sample(frac=self.sample_rate_dict.get(dirname, 1.0))
                
                # 分块写入
                for i in range(0, len(df), 128):  # 每128个样本一组
                    chunk = df.iloc[i:i+128]
                    buffer.append(chunk)
                    self._flush_buffer(buffer, self.prepare_output_dir)
                    buffer = []
            except Exception as e:
                print(f"Error processing {fpath}: {str(e)}")
        
        # 写入剩余数据
        if buffer:
            self._flush_buffer(buffer, self.prepare_output_dir)

    def _shuffle_stage(self):
        """混洗阶段：全局哈希混洗"""
        # 获取预处理后的文件列表
        if self.rank == 0:
            prepared_files = [
                f for f in self.fs.ls(self.prepare_output_dir)
                if f.endswith(".parquet")
            ]
            np.random.shuffle(prepared_files)
        else:
            prepared_files = None
        prepared_files = self.comm.bcast(prepared_files, root=0)
        my_files = prepared_files[self.rank::self.world_size]

        # 全局混洗处理
        global_buffer = []
        for fpath in tqdm(my_files, desc=f"Rank-{self.rank} Shuffling"):
            try:
                df = pq.read_table(fpath).to_pandas()
                df["_shard_hash"] = df.uuid.apply(hash)
                global_buffer.append(df)
                
                if len(global_buffer) * 128 >= self.buffer_mem_size:
                    merged = pd.concat(global_buffer)
                    merged.sort_values("_shard_hash", inplace=True)
                    self._write_shuffled(merged)
                    global_buffer = []
                    gc.collect()
            except Exception as e:
                print(f"Error shuffling {fpath}: {str(e)}")
        
        # 处理剩余数据
        if global_buffer:
            merged = pd.concat(global_buffer)
            merged.sort_values("_shard_hash", inplace=True)
            self._write_shuffled(merged)

    def _flush_buffer(self, buffer, output_dir):
        """写入缓冲数据"""
        if not buffer:
            return
        combined = pd.concat(buffer)
        filename = f"prep-{self.rank}-{uuid.uuid4().hex}.parquet"
        tmp_path = os.path.join(output_dir, ".tmp", filename)
        final_path = os.path.join(output_dir, filename)
        
        # 保证原子写入
        pq.write_table(pa.Table.from_pandas(combined), tmp_path)
        time.sleep(0.5)  # 防止HDFS元数据延迟
        self.fs.mv(tmp_path, final_path)

    def _write_shuffled(self, df):
        """写入混洗后的数据"""
        chunk_size = max(
            self.target_partition_size,
            len(df) // (len(df)//self.target_partition_size + 1)
        )
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i+chunk_size]
            filename = f"shuffle-{uuid.uuid4().hex}.parquet"
            tmp_path = os.path.join(self.final_output_dir, ".tmp", filename)
            final_path = os.path.join(self.final_output_dir, filename)
            
            pq.write_table(pa.Table.from_pandas(chunk), tmp_path)
            time.sleep(0.3)
            self.fs.mv(tmp_path, final_path)

    def run(self):
        # 阶段1：预处理
        self._prepare_stage()
        self.comm.barrier()  # 确保所有节点完成预处理
        
        # 阶段2：自动触发混洗
        self._shuffle_stage()
        self.comm.barrier()

def main():
    parser = argparse.ArgumentParser(description="Auto Two-Stage Shuffler")
    parser.add_argument("--input", required=True, nargs="+", 
                       help="输入目录，支持@采样率语法（如 /data@0.5）")
    parser.add_argument("--output", required=True, 
                       help="最终输出目录")
    parser.add_argument("--buffer", type=int, default=1024,
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
    shuffler.run()

if __name__ == "__main__":
    main()