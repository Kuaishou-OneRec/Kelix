import os
import json
import argparse
from mpi4py import MPI
from glob import glob
import pandas as pd
from tqdm import tqdm
import pyarrow.parquet as pq
import webdataset as wds
from omegaconf import OmegaConf
from recovlm.utils.blobstore_client import BlobStoreClient
from filters import create_filter
from data_sources import create_datasource

def get_parquet_filenames(dir):
    files = glob(os.path.join(dir, "*.parquet"))
    return sorted(files)

def process_single_shard(df, shard_id, output_folder, client):
    shard_name = "{shard_id:05d}".format(shard_id=shard_id)
    with wds.TarWriter(os.path.join(output_folder, f"{shard_name}.tar")) as tarwriter:
        pass

class Worker(object):

    def __init__(self, config):
        self.config = config
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.world_size = self.comm.Get_size()

        self.shard_size = config.shard_size
        self.txt_column = config.txt_column

        if self.rank == 0:
            if not os.path.exists(config.output_dir):
                os.makedirs(config.output_dir)
            self.files = get_parquet_filenames(config.meta_table)
        else:
            self.files = None
        self.files = self.comm.bcast(self.files, root=0)

        self.encode_format = self.config.encode_format
        self.data_source = create_datasource(config.data_source.class_name, config.data_source.kwargs)

        self.filters = [
            create_filter(c.class_name, c.kwargs) for c in config.filters
        ]

        # init writer
        self._shard_id = self.rank
        self._cur_size = 0
        self.tarwriter = None
        self.create_new_writer()
    
    def create_new_writer(self):
        if self.tarwriter is not None:
            print(f"{self._cur_fn} is written")
            self.tarwriter.close()
            self._cur_size = 0
            self._shard_id += self.world_size
        self._cur_fn = os.path.join(self.config.output_dir, f"{self._shard_id:05d}.tar")
        self.tarwriter = wds.TarWriter(self._cur_fn)
    
    def write_sample(self, sample):
        self.tarwriter.write(sample)
        self._cur_size += 1
        if self._cur_size >= self.shard_size:
            self.create_new_writer()
    
    def process_dataframe(self, df: pd.DataFrame):
        success = 0
        failed = 0
        for _, row in tqdm(df.iterrows(), desc=f"RANK[{self.rank:04d}]"):
            meta = row.to_dict()
            data = self.data_source(meta)
            if data is not None:
                success += 1
                sample = {
                    "__key__": f"{self._shard_id:06d}{self._cur_size:04d}",
                    "json": json.dumps(meta),
                    self.encode_format: data,
                }
                if self.txt_column is not None:
                    sample['txt'] = meta[self.txt_column]
                self.write_sample(sample)
            else:
                failed += 1
        print(f"RANK[{self.rank:04d}] success {success} failed {failed}")
    
    def run(self):
        print('self.files', self.files)
        for fn in tqdm(self.files):
            df = pq.read_table(fn, columns=list(self.config.meta_columns)).to_pandas()
            df = df[df.index % self.world_size == self.rank].copy()
            original_size = len(df)
            for f in self.filters:
                df = f(df)
            filtered_size = len(df)
            print(f"original_size is {original_size}, filtered_size is {filtered_size}")
            self.process_dataframe(df)
        self.tarwriter.close()

def main():
    parser = argparse.ArgumentParser(
        description="tool for download image from blobstore"
    )
    parser.add_argument("config_file")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config_file)
    print('config', cfg)
    worker = Worker(cfg)
    worker.run()

if __name__ == '__main__':
    main()