import os
import numpy as np
import os.path as osp
import pyarrow as pa
import subprocess
import glob
import logging
import shutil
import pyarrow.parquet as pq
logger = logging.getLogger(__name__)


class FakeParquetFileFromFastParquetFile:
    def __init__(self, fast_parquet_file):
        from fastparquet import ParquetFile
        self.fast_parquet_file = fast_parquet_file

        # 把打开文件逻辑放在前面，防止文件被删除而打开失败
        self.parquet = ParquetFile(self.fast_parquet_file)
        self.parquet.num_rows = len(self.parquet.to_pandas())
        self.num_row_groups = 1

    def read_row_group(self, i):
        assert i == 0
        return self.parquet


def load_parquet_file(fn: str, cache_dir: str, worker_id: int, rank_id: int, retry=5, max_cache_files=20, parquet_backend='fast_parquet') -> pq.ParquetFile:
    """
    加载 Parquet 文件，如果 HDFS 读取失败，则回退到本地缓存。

    Args:
        fn (str): Parquet 文件的路径，可以是 HDFS 路径
        cache_dir (str): 缓存路径
        worker_id (int): 工作进程id
        rank_id (int): 分布式rank id
        retry (int): 重试次数
        max_cache_files (int): 缓存中保留的最大文件数
        parquet_backend (str): Parquet 后端，可选 'fast_parquet' 或 'pyarrow'

    Returns:
        pq.ParquetFile: 加载的 Parquet 文件对象

    Raises:
        Exception: 如果 HDFS 和本地缓存加载都失败，则抛出异常
    """
    """Load a parquet file, with fallback to local cache if HDFS read fails.

    Args:
        fn (str): Path to parquet file, can be HDFS path
        retry (int): Number of retries
        max_cache_files (int): Maximum number of files to keep in cache

    Returns:
        pq.ParquetFile: Loaded parquet file object

    Raises:
        Exception: If both HDFS and local cache loading fail
    """
    import hashlib
    assert parquet_backend in ["fast_parquet", "pyarrow"]

    def calculate_text_hash(text):
        hash_object = hashlib.sha256()
        hash_object.update(text.encode('utf-8'))
        hash_hex = hash_object.hexdigest()
        return hash_hex

    cache_dir = osp.join(cache_dir, f'/{worker_id}_{rank_id}')
    os.makedirs(cache_dir, exist_ok=True)
    filename = osp.basename(fn)

    cache_fn = osp.join(cache_dir, str(calculate_text_hash(fn)) + '_' + filename)
    import time

    def clean_cache_if_needed():
        files = [_file for _file in glob.glob(osp.join(cache_dir, '*.parquet'), recursive=False) if osp.isfile(_file)]
        if len(files) > max_cache_files:
            files.sort(key=osp.getctime)
            for _file in files[:max_cache_files // 2]:
                os.remove(_file)
                logger.warning(f"Removing old cached file: {fn}")

    def read(_file):
        if parquet_backend == "pyarrow":
            parquet = pq.ParquetFile(_file)
        else:
            parquet = FakeParquetFileFromFastParquetFile(_file)

        return parquet

    for r in range(retry):
        logger.warning(f"retrying for fn={fn}/{cache_fn} {r} times.")
        try:
            if osp.exists(cache_fn):
                parquet_file = read(cache_fn)
                if parquet_file is not None:
                    return parquet_file
        except Exception as e:
            logger.warning(f"read failed for {r + 1} times. " + str(e))
        clean_cache_if_needed()
        try:
            command = f'/home/hadoop/software/hadoop/bin/hadoop fs -get {fn} {cache_fn}'
            subprocess.run(command, shell=True, check=True)
            parquet_file = read(cache_fn)
            if parquet_file is not None:
                return parquet_file
        except Exception as e:
            time.sleep(2 + np.random.randint(0, 5))
            logger.warning(f"download failed for {r + 1} times." + str(e))

    raise Exception(f"Failed to load parquet file from both original path and cache for {retry} times.")
