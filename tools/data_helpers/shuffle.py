import sys
import os
import os.path as osp
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


class Buffer(object):

    def __init__(self, buffer=None):
        self.buffer: pd.DataFrame = buffer

    def __len__(self):
        if self.buffer is None:
            return 0
        return len(self.buffer)

    def append(self, df):
        if self.buffer is None:
            self.buffer = df
            return
        self.buffer = pd.concat([self.buffer, df], ignore_index=True, axis=0)

    def pop_sample(self, n):
        if len(self) == 0:
            raise OverflowError("Buffer is empty.")
        df = self.buffer
        sample = df.sample(n=min(len(df), n))
        df = df.drop(sample.index)
        df = df.reset_index(drop=True)
        self.buffer = df
        return sample.reset_index(drop=True)


class BufferShuffler(MPIBase):

    def __init__(
            self,
            input_dir,
            output_dir,
            max_buffer_size=51200,
            out_partition=512,
    ):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.max_buffer_size = max_buffer_size
        self.out_partition = out_partition
        self.data = Buffer()
        self.tmp_dir = osp.join(self.output_dir, ".tmp")
        self.fs = pa.hdfs.connect(user="mpi")
        self.sample_rate_dict = dict()
        self.scatter_files = list()
        if self.rank == 0:
            files = []
            for d in tqdm(input_dir, desc="list directory"):
                if "@" in d:
                    d, sample_rate = d.strip().split("@")
                    self.sample_rate_dict[d.rstrip("/")] = float(sample_rate)
                files.extend(self.fs.ls(d))
            files = [
                x for x in files
                if "parquet" in x
            ]
            np.random.shuffle(files)
            while len(files) % self.world_size != 0:
                files.append(None)

            self.fs.mkdir(self.output_dir)
            self.fs.mkdir(self.tmp_dir)
        else:
            files = None
        files = self.comm.bcast(files, root=0)
        self.files = files[self.rank::self.world_size]
        self.sample_rate_dict = self.comm.bcast(self.sample_rate_dict, root=0)

    def write_df(self, df):
        basename = f"rank-{self.rank}-{str(uuid.uuid1())}.parquet"
        tmp_file = osp.join(self.tmp_dir, basename)
        filename = osp.join(self.output_dir, basename)
        pq.write_table(pa.Table.from_pandas(df), tmp_file)
        time.sleep(0.1)
        self.fs.mv(tmp_file, filename)
        self.mpi_print(f"write to {filename} success")

    def flush(self, threshold: int):
        data = self.data
        while len(data) > threshold:
            sample = data.pop_sample(self.out_partition)
            self.write_df(sample)

    def alltoall_by_chunk(self, df, chunk_size, return_list=True):

        rank = self.rank
        world_size = self.world_size
        comm = self.comm

        df_list = list()
        for chunk_id in range(chunk_size):
            pdf = df.iloc[chunk_id::chunk_size].copy()
            send_dfs = [pdf[pdf["seed"] == r] for r in range(world_size)]

            recv_dfs = comm.alltoall(send_dfs)
            pdf = pd.concat(recv_dfs, axis=0, ignore_index=True)
            df_list.append(pdf)
        if return_list:
            return df_list
        return pd.concat(df_list, axis=0, ignore_index=True)

    def build_empty_df(self, df):
        rank = self.rank
        world_size = self.world_size
        comm = self.comm

        if rank == 0:
            assert df is not None
            empty = df.iloc[:1].copy()
            empty = empty.drop(empty.index)
        else:
            empty = None

        empty = comm.bcast(empty, root=0)
        if df is not None:
            return df
        return empty

    def process_df(self, df):
        df = self.build_empty_df(df)
        df["seed"] = list(np.random.choice(range(self.world_size), size=len(df), replace=True))
        df = self.alltoall_by_chunk(df, chunk_size=1, return_list=False)
        df = df.drop(columns=["seed"])
        return df

    def run(self):

        data = self.data
        for fn in tqdm(self.files):
            if fn is not None:
                dirname = osp.dirname(fn).rstrip("/")
                sample_rate = self.sample_rate_dict.get(dirname, 1.0)
                df = pq.read_table(fn).to_pandas()
                df = df.sample(frac=sample_rate)
            else:
                df = None
            df = self.process_df(df)
            data.append(df)
            self.flush(self.max_buffer_size)

        self.flush(0)
        self.comm.barrier()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, nargs="*", required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_buffer_size", type=int, default=51200)
    parser.add_argument("--out_partition", type=int, default=512)
    args = parser.parse_args()
    worker = BufferShuffler(args.input_dir, args.output_dir, args.max_buffer_size, args.out_partition)
    worker.run()


if __name__ == '__main__':
    main()