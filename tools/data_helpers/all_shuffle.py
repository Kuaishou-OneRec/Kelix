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

class Shuffler(MPIBase):

    def __init__(
        self, 
        input_dir, 
        output_dir,
        buffer_mem_size=14*1024*1024*1024,
        out_partition=2048,
    ):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.buffer_mem_size = buffer_mem_size
        self.out_partition = out_partition
        self.tmp_dir = os.path.join(self.output_dir, ".tmp")
        self.fs = pa.hdfs.connect(user="mpi")
        self.sample_rate_dict = dict()
        if self.rank == 0:
            files = []
            for d in tqdm(input_dir, desc="list directory"):
                if "@" in d:
                    d, sample_rate = d.strip().split("@")
                    self.sample_rate_dict[d] = float(sample_rate)
                files.extend(self.fs.ls(d))
            files = [
                x for x in files
                if "parquet" in x
            ]
            np.random.shuffle(files)
            self.fs.mkdir(self.output_dir)
            self.fs.mkdir(self.tmp_dir)
        else:
            files = None
        files = self.comm.bcast(files, root=0)
        self.files = files[self.rank::self.world_size]
    
    def write_df(self, df):
        basename = f"rank-{self.rank}-{str(uuid.uuid1())}.parquet"
        tmp_file = os.path.join(self.tmp_dir, basename)
        filename = os.path.join(self.output_dir, basename)
        pq.write_table(pa.Table.from_pandas(df), tmp_file)
        time.sleep(0.1)
        self.fs.mv(tmp_file, filename)
        self.mpi_print(f"write to {filename} success")
    
    def shard_shuffle(self, df):
        df['hash'] = [hash(x) for x in df.uuid.values]
        df.sort_values(by='hash', inplace=True)
        for i in range(0, len(df), self.out_partition):
            cur_df = df.iloc[i:i+self.out_partition]
            try:
                self.write_df(cur_df)
            except Exception as e:
                print(traceback.format_exc())
        gc.collect()
    
    def run(self):
        buffer = []
        mem_size = 0
        for fn in tqdm(self.files):
            try:
                dirname = os.path.dirname(fn)
                sample_rate = self.sample_rate_dict.get(dirname, 1.0)
                df = pq.read_table(fn).to_pandas()
                df = df.sample(frac=sample_rate)
            except Exception as e:
                print(f"read {fn} error {e}")
                print(traceback.format_exc())
            buffer.append(df)
            mem_size += sys.getsizeof(df)
            if mem_size >= self.buffer_mem_size:
                df = pd.concat(buffer)
                buffer = []
                mem_size = 0
                self.shard_shuffle(df)

        df = pd.concat(buffer)
        buffer = []
        mem_size = 0
        self.shard_shuffle(df)
        self.comm.barrier()
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, nargs="*", required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--buffer_mem_size", type=int, default=8*1024*1024*1024)
    parser.add_argument("--out_partition", type=int, default=2048)
    args = parser.parse_args()
    worker = Shuffler(args.input_dir, args.output_dir, args.buffer_mem_size, args.out_partition)
    worker.run()

if __name__ == '__main__':
    main()