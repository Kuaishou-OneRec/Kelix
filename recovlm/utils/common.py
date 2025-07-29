# TODO: clean utils
from typing import Tuple
from rich import print
import time
import torch
import random
import numpy as np
from pathlib import Path
from transformers import set_seed as set_transformers_seed
import torch.distributed as dist
import pickle
import traceback
import subprocess
import os
try:
    from infra.perflog import create_perf_context
    INFRA_AVAILABLE = True
except ImportError:
    INFRA_AVAILABLE = False
    print("Warning: infra module not available, heart_beat functionality will be disabled")
import pyarrow.parquet as pq

def print_rank_n(*msg, rank=0):
  if dist.get_rank() == rank:
    print(*msg)

def print_rank_0(*msg):
  print_rank_n(*msg, rank=0)

def get_optimizer_grouped_parameters(model,
                                     learning_rate: float,
                                     vision_learning_rate: float,
                                     weight_decay,
                                     no_decay_name_list=[
                                         "bias", "LayerNorm.weight"
                                     ],
                                     vision_learning_rate_layer_dacay=1.0):
  optimizer_grouped_parameters = []

  llm_wd_params_group = []
  llm_nowd_params_group = []
  vit_wd_params_group = []
  vit_nowd_params_group = []

  for n, p in model.named_parameters():
    if p.requires_grad:
      if any(nd in n for nd in no_decay_name_list):
        # no weight decay params
        if n.startswith("visual"):
          vit_nowd_params_group.append((n, p))
        else:
          llm_nowd_params_group.append((n, p))
      else:
        # weight decay params
        if n.startswith("visual"):
          vit_wd_params_group.append((n, p))
        else:
          llm_wd_params_group.append((n, p))
  
  # for LLM
  optimizer_grouped_parameters.append({
    "params": [p for n, p in llm_wd_params_group],
    "weight_decay": 0.0,
    "lr": learning_rate,
  })
  optimizer_grouped_parameters.append({
    "params": [p for n, p in llm_nowd_params_group],
    "weight_decay": weight_decay,
    "lr": learning_rate,
  })

  # for vit
  if vision_learning_rate_layer_dacay == 1.0:
    optimizer_grouped_parameters.append({
      "params": [p for n, p in vit_wd_params_group],
      "weight_decay": weight_decay,
      "lr": vision_learning_rate,
    })
    optimizer_grouped_parameters.append({
      "params": [p for n, p in vit_nowd_params_group],
      "weight_decay": 0.0,
      "lr": vision_learning_rate,
    })
  else:
    # decay by layer
    vit_opt_groups = []

    # get all vit layers
    layer_ids = set()
    for n, _ in vit_wd_params_group + vit_nowd_params_group:
      if n.startswith("visual.blocks."):
        layer_ids.add(int(n.split(".")[2]))
    layer_ids = list(layer_ids)
    layer_ids.sort()
    layer_ids.reverse()
    # cal layer lr
    layers_params = []
    for idx, lid in enumerate(layer_ids):
      layer_lr = vision_learning_rate * (vision_learning_rate_layer_dacay ** idx)
      print_rank_0(f"visual.blocks.{lid}. {layer_lr=}")

      layers_params.extend([n for n, p in vit_wd_params_group if n.startswith(f"visual.blocks.{lid}.")])
      layers_params.extend([n for n, p in vit_nowd_params_group if n.startswith(f"visual.blocks.{lid}.")])
      
      vit_opt_groups.append({
        "params": [p for n, p in vit_wd_params_group if n.startswith(f"visual.blocks.{lid}.")],
        "weight_decay": weight_decay,
        "lr": layer_lr,
      })
      vit_opt_groups.append({
        "params": [p for n, p in vit_nowd_params_group if n.startswith(f"visual.blocks.{lid}.")],
        "weight_decay": 0.0,
        "lr": layer_lr,
      })

    optimizer_grouped_parameters.append({
        "params": [p for n, p in vit_wd_params_group if n not in layers_params],
        "weight_decay": weight_decay,
        "lr": vision_learning_rate,
    })
    optimizer_grouped_parameters.append({
        "params": [p for n, p in vit_nowd_params_group if n not in layers_params],
        "weight_decay": 0.0,
        "lr": vision_learning_rate,
    })
    optimizer_grouped_parameters.extend(vit_opt_groups)
    
  # for n, _ in llm_wd_params_group:
  #   print_rank_0(f"llm_weight_decay_params: {n}")
  # for n, _ in llm_nowd_params_group:
  #   print_rank_0(f"llm_no_weight_decay_params: {n}")
  # for n, _ in vit_wd_params_group:
  #   print_rank_0(f"vit_weight_decay_params: {n}")
  # for n, _ in vit_nowd_params_group:
  #   print_rank_0(f"vit_no_weight_decay_params: {n}")
  
  # remove empty params group
  final_optimizer_grouped_parameters = []
  for group in optimizer_grouped_parameters:
    if len(group['params']) > 0:
      final_optimizer_grouped_parameters.append(group)
  return final_optimizer_grouped_parameters

def to_device(batch, device, non_blocking=True):
  for key in list(batch.keys()):
    if isinstance(batch[key], torch.Tensor):
      batch[key] = batch[key].to(device=device, non_blocking=non_blocking)
  return batch

def to_cuda(batch, non_blocking=True):
  to_device(batch, device=torch.cuda.current_device(), non_blocking=non_blocking)

def set_random_seed(seed):
  if seed is not None:
    set_transformers_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def increment_version(version):
  major, minor, patch = map(int, version.split('.'))
  patch += 1
  return f"{major}.{minor}.{patch}"

def dist_reduce_dict(local_dict, group=None):
  gather_list = [None for _ in range(dist.get_world_size(group=group))]

  dist.all_gather_object(
    object_list=gather_list, obj=local_dict, group=group)

  def reduce_dicts(dicts):
    def _reduce(d1, d2):
      for key, value in d2.items():
        if isinstance(value, dict):
          if key not in d1:
            d1[key] = {}
          _reduce(d1[key], value)
        else:
          if key in d1:
            d1[key] += value
          else:
            d1[key] = value
      return d1

    result = {}
    for d in dicts:
        result = _reduce(result, d)
    return result

  return reduce_dicts(gather_list)

class Timer:
  def __init__(self, desc: str = ""):
    self.desc = desc

  def __enter__(self):
    print_rank_0(f"Start... {self.desc}")
    self.start = time.time()
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self.end = time.time()
    self.elapsed = self.end - self.start
    print_rank_0(f"End... {self.desc} elapsed: {self.elapsed:.3f} ")

def shell_hdfs_ls(source_dir):
  try:
    command = f"hdfs dfs -ls {source_dir}"
    result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
    files = []
    for line in result.stdout.splitlines():
      parts = line.split()
      if len(parts) > 0 and parts[-1].startswith('viewfs://'):
        files.append(parts[-1])
    return files

  except subprocess.CalledProcessError as e:
    print(f"Error occurred: {traceback.format_exc()}")
    return []

def pytorch_worker_info(group=None):  # sourcery skip: use-contextlib-suppress
  """Return node and worker info for PyTorch and some distributed environments.

  Args:
      group (optional): The process group for distributed environments. Defaults to None.

  Returns:
      tuple: A tuple containing (rank, world_size, worker, num_workers).
  """
  rank = 0
  world_size = 1
  worker = 0
  num_workers = 1
  if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
  else:
    try:
      import torch.distributed

      if torch.distributed.is_available() and torch.distributed.is_initialized():
        group = group or torch.distributed.group.WORLD
        rank = torch.distributed.get_rank(group=group)
        world_size = torch.distributed.get_world_size(group=group)
    except ModuleNotFoundError:
      pass
  if "WORKER" in os.environ and "NUM_WORKERS" in os.environ:
    worker = int(os.environ["WORKER"])
    num_workers = int(os.environ["NUM_WORKERS"])
  else:
    try:
      import torch.utils.data

      worker_info = torch.utils.data.get_worker_info()
      if worker_info is not None:
        worker = worker_info.id
        num_workers = worker_info.num_workers
    except ModuleNotFoundError:
      pass

  return rank, world_size, worker, num_workers

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

def get_task_tag():
    kml_id = os.environ.get("KML_ID", "")
    kml_task_id = os.environ.get("KML_TASK_ID", "")
    if kml_id.strip() == "" or kml_task_id.strip() == "":
        raise ValueError(f"env not set!!!! KML_ID={kml_id} KML_TASK={kml_task_id}")
    task_tag = f"kml-task-{kml_task_id}-record-{kml_id}"
    return task_tag

def heart_beat(num_tokens):
    if not INFRA_AVAILABLE:
        return
    current_file = os.path.abspath(__file__)
    log_ctx = create_perf_context('reco_vllm.pretrain', get_task_tag(), biz_def='infra', extra1=current_file)
    log_ctx.logstash_only(count=num_tokens)
    log_ctx.persist_data()

def get_root_dir():
  current_file_path = Path(__file__).resolve()
  # recovlm/recovlm/utils/common.py
  root_dir = current_file_path.parent.parent.parent
  return root_dir

def load_env():
  env_path = get_root_dir() / ".deepspeed_env"
  env = {}
  with open(env_path, encoding="utf-8") as f:
    for line in f:
      key, value = line.strip().split("=")
      env[key] = str(value)
  return env



def get_next_free_port(start_port):
  port = start_port
  while port <= 65535:  # 端口号最大为 65535
      try:
          # 创建一个 TCP socket
          sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
          # 尝试绑定到指定的端口
          sock.bind(('0.0.0.0', port))
          # 关闭 socket
          sock.close()
          return port
      except OSError:
          # 如果端口被占用，继续尝试下一个端口
          port += 1
  return None  # 如果没有找到可用端口，返回 None


import hashlib

def calculate_text_hash(text):
    # 创建一个 SHA-256 哈希对象
    hash_object = hashlib.sha256()
    # 将文本编码为字节串并更新哈希对象
    hash_object.update(text.encode('utf-8'))
    # 获取十六进制表示的哈希值
    hash_hex = hash_object.hexdigest()
    return hash_hex

def get_world_size_and_rank() -> Tuple[int, int]:
    """Function that gets the current world size (aka total number
    of ranks) and rank number of the current process in the default process group.

    Returns:
        Tuple[int, int]: world size, rank
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size(), torch.distributed.get_rank()
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"]), int(os.environ["RANK"])
    else:
        return 1, 0


class FakeParquetFileFromFastParquetFile:
    def __init__(self, fast_parquet_file):
        # 包的版本： mpirun --allow-run-as-root --hostfile /etc/mpi/hostfile --pernode bash -c "pip3 install fastparquet==2024.2.0"
        from fastparquet import ParquetFile
        self.fast_parquet_file = fast_parquet_file

        # 把打开文件逻辑放在前面，防止文件被删除而打开失败
        self.res = ParquetFile(self.fast_parquet_file)
        self.res.num_rows = len(self.res.to_pandas())
        self.num_row_groups = 1

    def read_row_group(self, i):
        assert i == 0
        return self.res


def load_parquet_file(fn: str, retry=5, max_cache_files=5, parquet_backend='fast_parquet') -> pq.ParquetFile:
    """
    加载 Parquet 文件，如果 HDFS 读取失败，则回退到本地缓存。

    Args:
        fn (str): Parquet 文件的路径，可以是 HDFS 路径
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
    if os.path.exists(fn):
      return  pq.ParquetFile(fn) if parquet_backend == 'pyarrow' else FakeParquetFileFromFastParquetFile(fn)

    def calculate_text_hash(text):
        # 创建一个 SHA-256 哈希对象
        hash_object = hashlib.sha256()
        # 将文本编码为字节串并更新哈希对象
        hash_object.update(text.encode('utf-8'))
        # 获取十六进制表示的哈希值
        hash_hex = hash_object.hexdigest()
        return hash_hex

    worker_id = get_worker_info()[0]
    rank_id = get_world_size_and_rank()[1]

    cache_dir = f'/code/dataset_cache/{worker_id}_{rank_id}'
    os.makedirs(cache_dir, exist_ok=True)
    filename = os.path.basename(fn)

    cache_fn = os.path.join(cache_dir, str(calculate_text_hash(fn)) + '_' + filename)
    import time

    def clean_cache_if_needed():
        files = [os.path.join(cache_dir, f) for f in os.listdir(cache_dir) if os.path.isfile(os.path.join(cache_dir, f))]
        if len(files) > max_cache_files:
            files.sort(key=os.path.getctime)
            for fn in files[:max_cache_files//2]:
                try: 
                  print(f"Removing old cached file: {fn}")
                  os.remove(fn)
                except: 
                  print(f"Failed to remove old cached file: {fn}")

    for r in range(retry):
        print(f"retrying for fn={fn}/{cache_fn}")
        try:
            if os.path.exists(cache_fn):
                res = pq.ParquetFile(cache_fn) if parquet_backend == 'pyarrow' else FakeParquetFileFromFastParquetFile(cache_fn)
            else:
                raise Exception(f"File not found on rank{dist.get_rank()}") # 直接用shell的方式
                # res = pq.ParquetFile(fn)
            return res
        
        except Exception as e:          
            # Try to download from HDFS
            try:
                clean_cache_if_needed()  # Clean cache before downloading new file
                cmd = f'/home/hadoop/software/hadoop/bin/hadoop fs -get {fn} {cache_fn}'
                os.system(cmd)
                res = pq.ParquetFile(cache_fn)  if parquet_backend == 'pyarrow' else FakeParquetFileFromFastParquetFile(cache_fn)
                return res
            except Exception as e2:
                time.sleep(2 + np.random.randint(0, 5))
                
                if r == retry - 1:
                    import traceback
                    traceback.print_exc()
                    raise Exception(f"Failed to load parquet file from both original path and cache.\nOriginal error: {e}\nCache error: {e2}")
