import os
from glob import glob
import pandas as pd
from tools.data_helpers.utils import MPIBase
from torch.utils.data import IterableDataset
from .dataset import DistDataset

class TheCaulDronDataset(DistDataset):

    def __init__(self, path):
        super().__init__()
        self.files = self.get_all_parquet_files(path)

    def get_all_parquet_files(self, path):
        if self.rank == 0:
            files = sorted(glob(os.path.join(path, "*/*.parquet")))
        else:
            files = None
        files = self.comm.bcast(files, root=0)
        files = files[self.rank::self.world_size]
        self.mpi_print(f"files number {len(files)}")
        return files
    
    def process_single_file(self, fn):
        df = pd.read_parquet(fn)
        for _, row in df.iterrows():
            row = row.to_dict()
            for text in row['texts']:
                s = dict()
                s['images'] = row['images']
                s['text'] = text
                yield s
    
    def __iter__(self):
        for fn in self.files:
            for s in self.process_single_file(fn):
                yield s
