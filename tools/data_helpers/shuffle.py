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
            self.buffer = df.reset_index(drop=True)
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

    def run(self):
        data = self.data
        for fn in tqdm(self.files):
            dirname = osp.dirname(fn).rstrip("/")
            sample_rate = self.sample_rate_dict.get(dirname, 1.0)
            df = pq.read_table(fn).to_pandas()
            df = df.sample(frac=sample_rate)
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
