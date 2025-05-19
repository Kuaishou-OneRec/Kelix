import os
import json
from glob import glob
import pyarrow as pa
import pyarrow.parquet as pq
from torch.utils.data import IterableDataset
import webdataset as wds
import tarfile
import base64
import pandas as pd
import traceback

class ParquetDataset(IterableDataset):
    def __init__(self, path, columns=None, user='mpi', limit=1000):
        super().__init__()
        self.path = path
        self.columns = columns
        self.user = user
        self.limit = limit
        self.shard_files = self.get_shard_files(path)

    def get_shard_files(self, path: str):
        # 如果路径以 viewfs:// 或 hdfs:// 开头，则采用 pa.hdfs.connect 获取文件列表
        if path.startswith("viewfs") or path.startswith("hdfs://"):
            self.fs = pa.hdfs.connect(user=self.user)
            files = self.fs.ls(path)
            # 筛选出以 parquet 结尾的文件，并进行排序
            files = sorted([x for x in files if x.endswith("parquet")])
            if not files:
                raise ValueError(f"No parquet files found in {path}")
            # 返回的 tuple 格式保持与原有代码一致： (filename, shard_id, shard_size)
            return [(fn, 0, 1) for fn in files]
        else:
            # 本地路径：首先判断是否为文件，如果是文件，直接返回；若为文件夹，则使用 glob 匹配 *.parquet
            if os.path.isfile(path):
                return [(path, 0, 1)]
            elif os.path.isdir(path):
                files = sorted(glob(os.path.join(path, "*.parquet")))
                if not files:
                    raise ValueError(f"No parquet files found in directory {path}")
                return [(fn, 0, 1) for fn in files]
            else:
                raise ValueError(f"Unsupported local path: {path}")

    def __iter__(self):
        count = 0
        for fn, _, _ in self.shard_files:
            try:
                if self.path.startswith("viewfs") or self.path.startswith("hdfs://"):
                    print(f"Reading parquet file (HDFS): {fn}")
                    with self.fs.open(fn, 'rb') as f:
                        parquet_file = pq.ParquetFile(f)
                        read_columns = self.columns if self.columns else None
                        for batch in parquet_file.iter_batches(batch_size=min(1000, self.limit or 1000)):
                            df = batch.to_pandas()
                            if read_columns:
                                df = df[read_columns]
                            for _, row in df.iterrows():
                                try:
                                    row_dict = {}
                                    for col, val in row.items():
                                        if col in ['images', 'messages', 'videos', 'segments']:
                                            if isinstance(val, str):
                                                try:
                                                    row_dict[col] = json.loads(val)
                                                except json.JSONDecodeError:
                                                    print(f"JSON decode error for column {col}: {val}")
                                                    row_dict[col] = val
                                            else:
                                                row_dict[col] = val
                                        else:
                                            row_dict[col] = val
                                    if row_dict:
                                        yield row_dict
                                        count += 1
                                        if self.limit and count >= self.limit:
                                            return
                                except Exception as e:
                                    print("Full traceback:")
                                    traceback.print_exc()
                                    print(f"Error processing row: {e}")
                                    continue
                else:
                    print(f"Reading parquet file: {fn}")
                    parquet_file = pq.ParquetFile(fn)
                    read_columns = self.columns if self.columns else None
                    for batch in parquet_file.iter_batches(batch_size=min(1000, self.limit or 1000)):
                        df = batch.to_pandas()
                        if read_columns:
                            df = df[read_columns]
                        for _, row in df.iterrows():
                            try:
                                row_dict = {}
                                for col, val in row.items():
                                    if pd.isna(val):
                                        continue
                                    if col in ['images', 'messages', 'videos', 'segments']:
                                        if isinstance(val, str):
                                            try:
                                                row_dict[col] = json.loads(val)
                                            except json.JSONDecodeError:
                                                print(f"JSON decode error for column {col}: {val}")
                                                row_dict[col] = val
                                        else:
                                            row_dict[col] = val
                                    else:
                                        row_dict[col] = val
                                if row_dict:
                                    yield row_dict
                                    count += 1
                                    if self.limit and count >= self.limit:
                                        return
                            except Exception as e:
                                print("Full traceback:")
                                traceback.print_exc()
                                print(f"Error processing row: {e}")
                                continue
            except Exception as e:
                print(f"Error processing file {fn}: {e}")
                traceback.print_exc()
                continue

class JsonlDataset(IterableDataset):
    def __init__(self, path):
        super().__init__()
        self.path = path
        with open(self.path, "r") as f:
            self.lines = f.readlines()
    
    def __iter__(self):
        for l in self.lines:
            if l.strip() != '':
                yield json.loads(l)

class JsonDataset(IterableDataset):
    def __init__(self, path):
        super().__init__()
        self.path = path
        with open(self.path, "r") as f:
            self.data = json.load(f)
    
    def __iter__(self):
        for s in self.data:
            yield s

class WebDataset(IterableDataset):
    def __init__(self, index_list):
        super().__init__()
        self.items = []
        for index_fn in index_list:
            with open(index_fn, encoding="utf-8") as f:
                index = json.loads(f.read())["shardlist"]
                for item in index:
                    item['url'] = os.path.join(
                        os.path.dirname(index_fn),
                        item['url']
                    )
                    self.items.append(item)
        self.total_samples = sum([x['nsamples'] for x in self.items])
    
    def __len__(self):
        return self.total_samples
    
    def __iter__(self):
        ds = wds.WebDataset(
            [x['url'] for x in self.items],
            handler=wds.warn_and_continue,
        )
        for s in ds:
            yield s

class TgzImageDataset(IterableDataset):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.fns = []
        self.data_cnt = 0
        fn_list = [fn for fn in os.listdir(path) if fn.endswith("tar.gz")]
        for fn in fn_list:
            self.fns.append(os.path.join(self.path, fn))
            with tarfile.open(os.path.join(self.path, fn), 'r:gz') as tar:
                self.data_cnt += len(tar.getnames())
    
    def __len__(self):
        return self.data_cnt
    
    def __iter__(self):
        for fn in self.fns:
            with tarfile.open(fn, 'r:gz') as tar:
                for member in tar.getmembers():
                    file = tar.extractfile(member)
                    name = member.name
                    if file is not None:
                        file_bytes = file.read()
                        image_sample = (name, base64.b64encode(file_bytes).decode('ascii'))
                        yield image_sample

class VlmTextJsonl(IterableDataset):
    def __init__(self, path):
        super().__init__()
        self.path = path
        with open(self.path, "r") as f:
            self.lines = f.readlines()
    
    def __len__(self):
        return len(self.lines)
    
    def __iter__(self):
        for l in self.lines:
            if l.strip() != '':
                src = json.loads(l)
                if src["images"] is not None and len(src["images"]) > 0:
                    pass
                else:
                    yield src
