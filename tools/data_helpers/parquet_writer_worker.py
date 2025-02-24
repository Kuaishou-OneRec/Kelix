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
from filters import create_filter
from typing import Dict, List, Sequence, Optional
import gc

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

        self._converters = []
        if "converter" in config:
            self._converters.append(
                create_converter(config.converter)
            )
        if "converters" in config:
            for cfg in config['converters']:
                self._converters.append(
                    create_converter(cfg)
                )

        self._pre_filters = []
        if "pre_filters" in config:
            for cfg in config['pre_filters']:
                self._pre_filters.append(self.create_filter(cfg))
        
        self._post_filters = []
        if "post_filters" in config:
            for cfg in config['post_filters']:
                self._post_filters.append(self.create_filter(cfg))
        
        self._filtered_cnt = 0
        self._success_cnt = 0
        self._none_cnt = 0
        self._filter_reason = dict()
    
    def create_filter(self, cfg):
        return self.filter_wrapper(cfg.class_name, create_filter(cfg))
    
    def filter_wrapper(self, name, filter_func):
        def f(x):
            rst = filter_func(x)
            if not rst:
                self._filter_reason[name] = self._filter_reason.get(name, 0) + 1
            return rst
        return f
    
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
        self._success_cnt += 1
        if self._buffer_size >= self.split_size:
            self.flush()

    def flush(self):
        if self._buffer_size > 0:
            tempfile = os.path.join(self.temp_dir, f"rank-{self.rank}-{str(uuid.uuid1())}.parquet")
            filename = os.path.join(self.output_dir, f"rank-{self.rank}-{str(uuid.uuid1())}.parquet")
            df = pd.DataFrame(self._buffer)
            self._buffer = []
            self._buffer_size = 0
            pq.write_table(pa.Table.from_pandas(df, nthreads=1), tempfile)
            self.fs.mv(tempfile, filename)
            self.mpi_print(f"write to {filename} success, total filtered_cnt {self._filtered_cnt} success_cnt {self._success_cnt}")
            self.mpi_print(f"filter_reason {self._filter_reason}")
        gc.collect()


    def run(self):
        # 获取并打印数据集总数
        total_rows = None if not hasattr(self.dataset, "__len__") else len(self.dataset)
        
        for s in tqdm(self.dataset, total=total_rows):
            try:
                if not all([f(s) for f in self._pre_filters]):
                    self._filtered_cnt += 1
                    continue
                out = s
                for cvt in self._converters:
                    out = cvt(out)
                if not all([f(s) for f in self._post_filters]):
                    self._filtered_cnt += 1
                    continue
                if out is not None:
                    if isinstance(out, Dict):
                        self.write_sample(out)
                    elif isinstance(out, Sequence):
                        for s in out:
                            self.write_sample(s)
                else:
                    self._none_cnt += 1
            except Exception as e:
                print(traceback.format_exc())
        self.flush()
        self.comm.barrier()
        
        # 打印处理成功的数据条数
        self.mpi_print(f"Total processed success count: {self._success_cnt}, total rows {total_rows}, filtered_cnt {self._filtered_cnt}, none_cnt {self._none_cnt}")

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

if __name__ == "__main__":
    main()