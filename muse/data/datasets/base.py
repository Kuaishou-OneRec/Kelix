from typing import Dict, Any, List, Optional, Union, Tuple
import random
import json
import os
import time
import torch
import hashlib
import base64
import traceback
import numpy as np
from PIL import Image
from io import BytesIO
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import IterableDataset
from fastparquet import ParquetFile
from torch.utils.data import DataLoader

PARQUET_CACHE_DIR = os.environ.get("PARQUET_CACHE_DIR", "/code/dataset_cache")

def is_hdfs(path: str) -> bool:
  """Check if a path is a HDFS path"""
  return path.startswith('viewfs://') or path.startswith('hdfs://')

def get_worker_info():
  worker = 0
  num_workers = 1
  try:
    import torch.utils.data

    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
      worker = worker_info.id
      num_workers = worker_info.num_workers
  except ModuleNotFoundError:
    pass
  return worker, num_workers

def get_world_size_and_rank():
  rank = os.environ.get("RANK", 0)
  world_size = os.environ.get("WORLD_SIZE", 1)
  try:
    import torch.distributed as dist
    rank = dist.get_rank()
    world_size = dist.get_world_size()
  except:
    pass
  return rank, world_size

def is_image_exist(image_path: str) -> bool:
  return image_path and os.path.exists(image_path) \
    and os.path.getsize(image_path) > 0

def load_image(image: str) -> Optional[Image.Image]:
  try:
    if not is_image_exist(image):
      image_bytes = base64.b64decode(image)
      image = Image.open(BytesIO(image_bytes))
    else:
      image = Image.open(image)
    return image
  except (ValueError, OSError, IOError) as e:
    print(f"Error loading image: {e}")
    return None

def rename(src: str, dst: str):
  if is_hdfs(src) and is_hdfs(dst):
    cmd = f'/home/hadoop/software/hadoop/bin/hadoop fs -mv {src} {dst}'
    os.system(cmd)
  else:
    os.system(f"mv {src} {dst}")

def calculate_text_hash(text):
  hash_object = hashlib.sha256()
  hash_object.update(text.encode('utf-8'))
  return hash_object.hexdigest()

def load_parquet(path: str, rank: int = 0, worker: int = 0) -> ParquetFile:
  """Load a parquet file, with fallback to local cache if HDFS read fails."""
  retry: int = 10
  max_cache_files: int = 10

  cache_dir = f'/code/dataset_cache/{rank}_{worker}'
  os.makedirs(cache_dir, exist_ok=True)
  filename = os.path.basename(path)

  cache_fn = os.path.join(
    cache_dir, str(calculate_text_hash(path)) + '_' + filename)

  def clean_cache_if_needed():
    files = [
      os.path.join(cache_dir, f) for f in os.listdir(cache_dir) \
        if os.path.isfile(os.path.join(cache_dir, f))]
    if len(files) > max_cache_files:
      files.sort(key=os.path.getctime)
      file_to_remove = max(len(files) - max_cache_files, 0)
      for fn in files[:file_to_remove]:
        print(f"Removing old cached file: {fn}")
        os.remove(fn)

  for r in range(retry):
    clean_cache_if_needed()
    try:
      if os.path.exists(path):
        return ParquetFile(path)
      elif os.path.exists(cache_fn):
        return ParquetFile(cache_fn)
      else:
        raise FileNotFoundError("File not found")
    except (OSError, IOError, FileNotFoundError, RuntimeError) as e:
      try:
        cmd = f'/home/hadoop/software/hadoop/bin/hadoop fs -get {path} {cache_fn}'
        if os.system(cmd) != 0:
          raise RuntimeError("HDFS get command failed")
        return ParquetFile(cache_fn)
      except (OSError, IOError, RuntimeError) as e2:
        time.sleep(2 + np.random.randint(0, 5))
        if r == retry - 1:
          raise IOError('parquet', path, 
            f"Failed to load from both original path and cache. "
            f"Original error: {e}\nCache error: {e2}")
    print(f"Retrying for path={path} / {cache_fn}")

class ParquetDataset(IterableDataset):
    """
    IterableDataset for parquet files, consuming files in order.
    """
    def __init__(self, files, rank=0):
      self.rank = rank
      self.files = files

    def _parser(self, row, filename):
        key = "unknown"
        try:
            assert "messages" in row, "messages is not in row"
            messages = json.loads(row["messages"])

            images = row["images"]
            videos = row["videos"]
            source = row["source"]
            key = row["uuid"]

            samples = {
                "__key__": key,
                "__url__": filename,
                "raw": row,
                "json": {
                    "source": source,
                    "messages": messages,
                }
            }
            images = json.loads(images)
            videos = json.loads(videos)

            for img_key, image in images.items():
              if self.is_load_image:
                image = load_image(image)
              if image:
                samples[img_key] = image
            for vid_key, video in videos.items():
                samples[vid_key] = video

            return samples

        except Exception as e:
            print(
                f"ParquetDataset parse sample error, __url__={filename}, __key__={key},"
                f"err_msg={traceback.format_exc()}")
            return None

    def __iter__(self,):
        worker_id, _  = get_worker_info()
        for fn in tqdm(self.files):
            try:
                parquet_file = load_parquet(fn, rank=self.rank, worker=worker_id)
            except Exception as e:
                print(f"open parquet fail {fn=}, error_msg={traceback.format_exc()}")
                continue
            df = parquet_file.to_pandas()
            for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"{worker_id=} Reading {fn}"):
                try:
                    sample = self._parser(row, fn)
                    if sample is not None:
                        yield sample
                except Exception as e:
                    print(f"Error processing row {idx} in {fn}: {str(e)}")
                    continue

class DistributedDataset(IterableDataset):
  def __init__(self,
               rank: int = 0,
               world_size: int = 1,
               num_workers: int = 8,
               seed: int = 1024):
    self.rank = rank
    self.world_size = world_size
    self.num_workers = num_workers
    self.seed = seed
  
  def __iter__(self):
    raise NotImplementedError("Subclass must implement this method")

class FileBasedDataset(DistributedDataset):
  def __init__(
      self,
      sources: str,
      rank: int = 0,
      world_size: int = 1,
      num_workers: int=8,
      seed: int=1024,
      num_epochs: int=1,
      shard_by: str = "auto",
      reader: str = "parquet",
      **kwargs):
    """
    Distributed dataset supporting three sharding modes:
    - shard_by="auto": Auto-select (default), uses 'files' if num_files >= total_workers, else 'samples'
    - shard_by="files": Shard by files, each worker processes different files (high I/O efficiency, suitable for sufficient files)
    - shard_by="samples": Shard by samples, all workers read same files but process different samples (better load balancing, lower I/O efficiency)
    
    Recommended usage:
    - Local/single-node multi-GPU training: prefer "files" (avoid I/O contention)
    - Distributed training + HDFS storage: prefer "files" (save network bandwidth)
    - Very few files but many samples: use "samples"
    - Uncertain: use "auto" (default)
    """
    super().__init__(rank=rank, world_size=world_size, num_workers=num_workers, seed=seed)
    assert shard_by in ["auto", "files", "samples"], \
        f"shard_by must be 'auto', 'files' or 'samples', got {shard_by}"
    
    self.rng = random.Random(seed)
    self.num_epochs = num_epochs
    self.sources = sources
    self.shard_by = shard_by
    self.reader = reader
    self.kwargs = kwargs
    
    # Initialize attributes
    self._all_files = []
    self._actual_shard_by = shard_by

    self.dataset = self._build()

  def _load_file_list(self) -> List[str]:
    """Load file list"""
    files = []
    if isinstance(self.sources, list):
        files = self.sources
    elif self.sources.endswith(".json"):
      with open(self.sources, "r") as fp:
        files = json.loads(fp.read())
        files = sorted([
          fn for fn in files if fn.endswith(".parquet")])
    else:
      folder = Path(self.sources)
      files = list(map(str, folder.rglob("*.parquet")))
    
    return files
  
  def _get_reader_class(self):
    if self.reader == "parquet":
      return ParquetDataset
    else:
      raise ValueError(f"Unsupported reader: {self.reader}")

  def _build(self):
    """Build dataset based on shard_by mode"""
    files = self._load_file_list()
    assert len(files) > 0, f"No file found for rank{self.rank}"

    # Shuffle file list with fixed seed (all ranks and workers use same seed to ensure consistent order)
    files_rng = random.Random(self.rng.getstate()[1][0])  # Use initial seed
    files_rng.shuffle(files)

    # Calculate total parallelism
    total_workers = self.world_size * self.num_workers
    
    # Determine actual sharding mode
    actual_shard_by = self.shard_by
    
    if self.shard_by == "auto":
      # Auto-select mode
      if len(files) >= total_workers:
        actual_shard_by = "files"
        if self.rank == 0:
          print(f"📂 Auto-selected 'files' mode: {len(files)} files >= {total_workers} workers")
      else:
        actual_shard_by = "samples"
        if self.rank == 0:
          print(f"📊 Auto-selected 'samples' mode: {len(files)} files < {total_workers} workers")
    elif self.shard_by == "files" and len(files) < total_workers:
      # Force files mode but switch to samples when files are insufficient
      actual_shard_by = "samples"
      if self.rank == 0:
        print(
          f"⚠️  Warning: File count ({len(files)}) < total workers ({total_workers}), "
          f"auto-switching from 'files' to 'samples' mode")

    if actual_shard_by == "files":
      # Shard by files: each (rank, worker) combination processes different files
      # Calculate current worker's global index
      # Note: actual worker_id can only be obtained in __iter__, store all files here first
      # Actual sharding will be done in __iter__
      
      print(
        f"DistributedDataset [shard_by=files] "
        f"rank={self.rank}/{self.world_size}, num_workers={self.num_workers}, "
        f"total_files={len(files)}, total_parallel_workers={total_workers}")
      
      # Store all files, distribute by worker in __iter__
      self._all_files = files
      self._actual_shard_by = "files"
      dataset = None  # Lazy creation
    else:
      # Shard by samples: all ranks/workers process same files
      print(
        f"DistributedDataset [shard_by=samples] "
        f"rank={self.rank}/{self.world_size}, num_workers={self.num_workers}, "
        f"total_files={len(files)}")
      
      self._actual_shard_by = "samples"
      dataset = self._get_reader_class()(files, rank=self.rank)
    
    return dataset

  def process(self,
              sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    raise NotImplementedError("Subclass must implement this method")



  def __iter__(self):
    worker_id, num_workers = get_worker_info()
    
    if self._actual_shard_by == "files":
      # File sharding mode: each (rank, worker) combination processes different files
      # Calculate current worker's global index
      global_worker_id = self.rank * self.num_workers + worker_id
      total_workers = self.world_size * self.num_workers
      
      # Distribute files by global worker index
      files_for_this_worker = self._all_files[global_worker_id::total_workers]
      
      print(
        f"  Worker[rank={self.rank}, worker={worker_id}, global={global_worker_id}] "
        f"processing {len(files_for_this_worker)} files")
      
      # Create dataset for current worker
      dataset = self._get_reader_class()(
        files_for_this_worker, 
        rank=self.rank
      )
      
      # Process all samples sequentially (files already distributed)
      for sample in dataset:
        inputs = self.process(sample)
        yield inputs
    else:
      # Sample sharding mode: need to consider both rank and worker sharding
      for idx, sample in enumerate(self.dataset):
        if idx % self.world_size == self.rank and \
                idx % num_workers == worker_id:
          inputs = self.process(sample)
          yield inputs
