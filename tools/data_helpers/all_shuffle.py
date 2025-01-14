import os
import uuid
from tqdm import tqdm
import argparse
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from utils import MPIBase

class Shuffler(MPIBase):

    def __init__(
        self, 
        input_dir, 
        output_dir,
        buffer_mem_size=16*1024*1024*1024,
        shuffle_partition=64,
    ):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.buffer_mem_size = buffer_mem_size
        self.shuffle_partition = shuffle_partition
        self.tmp_dir = os.path.join(self.output_dir, ".tmp")
        self.fs = pa.hdfs.connect(user="mpi")
        if self.rank == 0:
            files = self.fs.ls(input_dir)
            files = [
                x for x in files
                if "parquet" in x
            ]
            self.fs.mkdir(self.output_dir)
            self.fs.mkdir(self.tmp_dir)
        else:
            files = None
        files = self.comm.bcast(files, root=0)
        self.files = files[self.rank::self.world_size]
    
    def write_df(self, df):
        for i in range(0, len(df), self.out_split_size):
            cur_df = df.iloc[i:i+self.out_split_size]
            basename = f"rank-{self.rank}-{str(uuid.uuid1())}.parquet"
            tmp_file = os.path.join(self.tmp_dir, basename)
            filename = os.path.join(self.output_dir, basename)
            pq.write_table(pa.Table.from_pandas(cur_df), tmp_file)
            self.fs.mv(tmp_file, filename)
            self.mpi_print(f"write to {filename} success")
    
    def shard_shuffle(self, df):
        df['hash'] = [hash(x) for x in df.uuid.values]
        for i in range(self.shuffle_partition):
            cur_df = df[df.hash % self.shuffle_partition == i]
            self.write_df(cur_df)
    
    def run(self):
        buffer = []
        mem_size = 0
        for fn in tqdm(self.files):
            df = pq.read_table(fn).to_pandas()
            buffer.append(df)
            mem_size += df.memory_usage().sum()
            if mem_size >= self.buffer_mem_size:
                df = pd.concat(buffer)
                buffer = []
                mem_size = 0
                self.shard_shuffle(df)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    worker = Shuffler(args.input_dir, args.output_dir)
    worker.run()

if __name__ == '__main__':
    main()