import os
import math
import json
from glob import glob
import pyarrow as pa
import pyarrow.parquet as pq
from tools.data_helpers.utils import MPIBase
from torch.utils.data import IterableDataset

def lcm(a: int, b: int):
    return a * b // (math.gcd(a, b))

class DistDataset(IterableDataset, MPIBase):
    def __init__(self):
        super().__init__()

class ParquetDataset(DistDataset):

    def __init__(self, path, columns = None, user='mpi'):
        super().__init__()
        self.path = path
        self.columns = list(columns)
        self.shard_files = self.get_shard_files(path)
    
    def get_shard_files(self, path: str):
        if self.rank == 0:
            if path.startswith("viewfs"):
                self.fs = pa.hdfs.connect(user=user)
                files = self.fs.ls(path)
                files = sorted([x for x in files if "SUCCESS" not in x])
            elif path.startswith("/"):
                files = sorted(glob(os.path.join(path, "*.parquet")))
            num_files = len(files)
            lcm_num_files_world_size = lcm(num_files, self.world_size)
            num_single_file_shard = lcm_num_files_world_size // num_files
            shard_files = [
                # filename, shard_id, shard_size
                (fn, sid, num_single_file_shard)
                for fn in files
                for sid in range(num_single_file_shard)
            ]
        else:
            shard_files = None
        shard_files = self.comm.bcast(shard_files, root=0)
        shard_files = shard_files[self.rank::self.world_size]
        self.mpi_print("shard_files", shard_files)
        return shard_files    
    
    def __iter__(self):
        for fn, sid, shard_size in self.shard_files:
            df = pq.read_table(fn, columns=self.columns).to_pandas()
            df = df[df.index % shard_size == sid]
            for _, row in df.iterrows():
                row = row.to_dict()
                yield row

class JsonlDataset(DistDataset):
    
    def __init__(self, path):
        super().__init__()
        self.path = path
        with open(self.path, "r") as f:
            lines = f.readlines()
        shard_size = len(lines) // self.world_size
        self.lines = lines[self.rank * shard_size: (self.rank + 1) * shard_size]
    
    def __iter__(self):
        for l in self.lines:
            if l.strip() != '':
                yield json.loads(l)

class JsonDataset(DistDataset):

    def __init__(self, path):
        super().__init__()
        self.path = path
        with open(self.path, "r") as f:
            self.data = json.load(f)
        self.data = self.data[self.rank::self.world_size]
    
    def __iter__(self):
        for s in self.data:
            yield s
