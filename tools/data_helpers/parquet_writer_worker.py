import sys
import os
import uuid
import argparse
import traceback
import pandas as pd
import pyarrow as pa
from tqdm import tqdm
import pyarrow.parquet as pq
from omegaconf import OmegaConf
from utils import MPIBase
from typing import Optional
from datasets import create_dataset
from converters import create_converter
from typing import Dict, List, Sequence, Optional

pa.jemalloc_set_decay_ms(0)

SCHEMA_DICT = {
    "images": str,
    "videos": list,
    "source": str,
    "messages": str,
    "segments": str,
    "metadata": str,
    "uuid": str,
}

class MPIParquetWriterWorker(MPIBase):

    def __init__(self, config):
        super().__init__()
        self.output_dir = config.output_dir
        self.temp_dir = os.path.join(self.output_dir, ".tmp")
        self.split_size = config.split_size
        self.fs = pa.hdfs.connect(user="mpi")
        if self.rank == 0:
            self.fs.mkdir(self.output_dir)
            self.fs.mkdir(self.temp_dir)
        self.comm.barrier()
    
        self._buffer = []
        self._buffer_size = 0

        self.dataset = create_dataset(config.dataset)
        self.converter = create_converter(config.converter)
    
    def is_sample_valid(self, sample):
        for k, t in SCHEMA_DICT.items():
            if k not in sample or not isinstance(sample[k], t):
                return False
        return True
    
    def sample_size(self, sample):
        rst = 0
        for k, v in sample.items():
            rst += sys.getsizeof(v)
        return rst
    
    def write_sample(self, sample):
        self._buffer.append(sample)
        self._buffer_size += self.sample_size(sample)
        if self._buffer_size >= self.split_size:
            self.flush()

    def flush(self):
        tempfile = os.path.join(self.temp_dir, f"rank-{self.rank}-{str(uuid.uuid1())}.parquet")
        filename = os.path.join(self.output_dir, f"rank-{self.rank}-{str(uuid.uuid1())}.parquet")
        df = pd.DataFrame(self._buffer)
        self._buffer = []
        self._buffer_size = 0
        pq.write_table(pa.Table.from_pandas(df, nthreads=1), tempfile)
        self.fs.mv(tempfile, filename)
        self.mpi_print(f"write to {filename} success")

    def run(self):
        for s in tqdm(self.dataset, total=len(self.dataset)):
            try:
                out = self.converter(s)
                if out is not None:
                    if isinstance(out, Dict):
                        self.write_sample(out)
                    elif isinstance(out, Sequence):
                        for s in out:
                            self.write_sample(s)
            except Exception as e:
                print(traceback.format_exc())

def main():
    parser = argparse.ArgumentParser(
        description="tool for convert dataset to parquet"
    )
    parser.add_argument("config_file")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config_file)
    print("config", cfg)
    worker = MPIParquetWriterWorker(cfg)
    worker.run()
    worker.flush()

if __name__ == "__main__":
    main()