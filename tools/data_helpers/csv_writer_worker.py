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

        if config.dataset.class_name == "csv":
            # 使用 Pandas 分块读取 CSV（支持大文件）
            self.csv_path = config.dataset.path
            self.chunksize = config.dataset.get("chunksize", 10000)
            self.df_iterator = pd.read_csv(
                self.csv_path, 
                chunksize=self.chunksize,
                iterator=True
            )
        else:
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
        # !!! 重构后的 CSV 读取逻辑
        total_rows = sum(1 for _ in open(self.csv_path)) - 1  # 估算总行数
    
        for chunk in self.df_iterator:  # 分块读取
            for _, row in tqdm(chunk.iterrows(), total=len(chunk)):
                try:
                    # !!! 将行数据转换为字典格式
                    sample = {col: row[col] for col in ['src_pid', 'src_caption', 'src_title', 'src_text', 'src_ocr', 'src_asr', 'sim_pid', 'sim_caption', 'sim_title', 'sim_text', 'sim_ocr', 'sim_asr', 'neg_pid', 'neg_caption', 'neg_title', 'neg_text', 'neg_ocr', 'neg_asr']}
                    out = sample
                    for cvt in self._converters:
                        out = cvt(out)
                
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