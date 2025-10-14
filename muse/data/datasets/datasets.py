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
from transformers import AutoTokenizer, AutoProcessor, AutoConfig
from fastparquet import ParquetFile
from evaluators.utils.qwen_vl_utils import process_vision_info as process_vision_info_qwen
from evaluators.utils.keye_vl_utils import process_vision_info as process_vision_info_keye
from torch.utils.data import DataLoader

from evaluators.prompts import PromptLoader

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
  except Exception as e:
    print(f"Error loading image: {e}")
    return None

def rename(src: str, dst: str):
  if is_hdfs(src) and is_hdfs(dst):
    cmd = f'/home/hadoop/software/hadoop/bin/hadoop fs -mv {src} {dst}'
    os.system(cmd)
  else:
    os.system(f"mv {src} {dst}")

def cleanup_cache():
  """Cleanup cache directory by force removing with system command"""
  _, rank = get_world_size_and_rank()
  worker, _ = get_worker_info()
  cache_dir = f'{PARQUET_CACHE_DIR}/{rank}_{worker}'

  # Force remove directory using system command
  os.system(f"rm -rf {cache_dir}")

  # Create fresh cache directory
  os.makedirs(cache_dir, exist_ok=True)

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
    except Exception as e:
      try:
        cmd = f'/home/hadoop/software/hadoop/bin/hadoop fs -get {path} {cache_fn}'
        if os.system(cmd) != 0:
          raise Exception("HDFS get command failed")
        return ParquetFile(cache_fn)
      except Exception as e2:
        time.sleep(2 + np.random.randint(0, 5))
        if r == retry - 1:
          raise IOError('parquet', path, 
            f"Failed to load from both original path and cache. "
            f"Original error: {e}\nCache error: {e2}")
    print(f"Retrying for path={path} / {cache_fn}")

class ParquetDataset(IterableDataset):
    """
    Parquet 数据集的通用基类，使用模板方法模式。
    """
    def __init__(self, files, is_load_image=True, rank=0):
        # 保序
        self.files = sorted(files)
        self.is_load_image = is_load_image
        self.rank = rank

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
  def __init__(
      self,
      sources: str,
      rank: int = 0,
      world_size: int = 1,
      num_workers: int=8,
      seed: int=1024,
      num_epochs: int=1,
      shard_by: str = "files",
      **kwargs):
    self.rng = random.Random(seed)
    self.rank = rank
    self.world_size = world_size
    self.num_workers = num_workers
    self.num_epochs = num_epochs
    self.sources = sources
    self.kwargs = kwargs
    self.preload = kwargs.get("preload", True)

    self.dataset = self._build()

  def _build(self):
    files = []
    # TODO: support more file types
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

    assert len(files) > 0, f"No file found for rank{self.rank}"

    print(
      f"DistributedDataset "
      f"rank={self.rank}, world_size={self.world_size} "
      f"file_num={len(files)}")

    dataset = ParquetDataset(files, is_load_image=self.is_load_image, rank=self.rank)
    return dataset

  def _shard_build(self):
    data_file_list = []
    if self.rank == 0:
      data_files = []
      if isinstance(sources, str) and sources.endswith(".json"):
        with open(sources, "r") as fp:
          data_files = json.loads(fp.read())
          data_files = [fn for fn in data_files if fn.endswith(".parquet")]
      elif isinstance(sources, list):
        for source in sources:
          hdfs_files = shell_hdfs_ls(source)
          data_files += [fn for fn in hdfs_files if fn.endswith(".parquet")]
      # repeat
      for i in range(self.num_epochs):
        data_files.sort()
        self.rng.shuffle(data_files)
        data_file_list += [(fn, i) for fn in data_files]
      print(
        f"Shard files rank{self.rank}: ori_file_num={len(data_files)} file_num={len(data_file_list)}")

    t = [data_file_list]
    dist.broadcast_object_list(t, src=0)
    data_file_list = t[0]


  def process(self,
              sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    raise NotImplementedError("Subclass must implement this method")

  def __iter__(self):
    # 样本sharding
    worker_id, num_workers = get_worker_info()
    for idx, sample in enumerate(self.dataset):
        if idx % self.world_size == self.rank and \
                idx % num_workers == worker_id:
            inputs = self.process(sample)
            yield inputs

class Qwen2VLDataset(DistributedDataset):
  "For vllm batch inference"
  def __init__(self,
               sources: Union[str, List[str]],
               rank: int = 0,
               world_size: int = 1,
               num_workers: int = 8,
               seed: int = 1024,
               system_prompt: str = "default",
               tokenizer_path: Optional[str] = None,
               **kwargs):
    super().__init__(
      sources=sources, rank=rank, world_size=world_size,
      num_workers=num_workers, seed=seed, **kwargs)
    prompt_loader = PromptLoader()
    self.system_prompt = prompt_loader.load(system_prompt)
    self.tokenizer_path = tokenizer_path
    self.min_pixels = \
      kwargs.get("min_pixels", 4 * 28 * 28)
    self.max_pixels = \
      kwargs.get("max_pixels", 512 * 28 * 28)
    self.max_images = kwargs.get("max_images", 10)
    self.fps = kwargs.get("fps", 1.0)
    self.nframes = kwargs.get("nframes", 0)
    self.min_frames = kwargs.get("min_frames", 1)
    self.max_frames = kwargs.get("max_frames", 10)
    self.processor = AutoProcessor.from_pretrained(tokenizer_path)

  def fill_image_block(self,
                       block: Dict[str, Any],
                       sample: Optional[Dict[str, Any]] = None):

    assert block["image"] in sample,\
      f"image key={block['image']} not in sample., {sample.keys()}"
    if block["image"] in sample:
      block["image"] = sample[block["image"]]
    else:
      if isinstance(block["image"], str):
        block["image"] = load_image(block["image"])
    # assert isinstance(block["image"], PIL.Image), "Failed to prepare image content"
    if block["image"].mode != "RGB":
      block["image"] = block["image"].convert("RGB")

    block["min_pixels"] = self.min_pixels
    block["max_pixels"] = self.max_pixels

    return block

  def fill_video_block(self,
                       block: Dict[str, Any],
                       sample: Dict[str, Any]):

    if isinstance(block["video"], list):
      # fake image block list，复用fill_image_block
      if all([isinstance(image_block, str) for image_block in block["video"]]):
        block["video"] = [{"type": "image", "image": image_str} for image_str in block["video"]]
      for block in block["video"]:
        self.fill_image_block(block, sample)

    elif isinstance(block["video"], str) or isinstance(block["video"], bytes):
      # video in local tar, replace by video bytes
      if isinstance(block["video"], str) and block["video"] in sample:
        block["video"] = sample[block["video"]]
      
      if isinstance(block["video"], str) and not os.path.exists(block["video"]):
        raise ValueError(f"video file not exists: {block['video']}")

      # fill other params
      block["min_pixels"] = self.min_pixels
      block["max_pixels"] = self.max_pixels
      # video split params
      if self.nframes > 0:
        block["nframes"] = self.nframes
      else:
        if self.fps > 0:
          block["fps"] = self.fps
        if self.min_frames > 0:
          block["min_frames"] = self.min_frames
        if self.max_frames > 0:
          block["max_frames"] = self.max_frames
    else:
      raise ValueError(
        f"Unsupport video type. {type(block['video'])=}")

    return block

  def process(self,
              sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    assert "messages" in sample["json"], \
        f"sample must contain `messages` key, but got {sample['json'].keys()}"

    messages = sample["json"]["messages"]
    prompt_messages = []
    for turn in messages:
      content = turn["content"]
      if isinstance(content, str):
        continue
      for block in content:
        if block["type"] == "image":
          self.fill_image_block(block, sample)
        elif block["type"] == "video":
          self.fill_video_block(block, sample)
        elif block["type"] == "text":
          continue
        else:
          raise ValueError(
            f"sample process error, unsupport value type: {block['type']}")

    if messages[0]["role"] != "system":
      system = {"role": "system", "content": self.system_prompt}
      messages.insert(0, system)

    for turn in messages:
      if turn["role"] == "assistant":
        break
      prompt_messages.append(turn)

    annotation = None
    if messages[-1]["role"] == "assistant":
      annotation = messages[-1]["content"]
      if isinstance(annotation, list):
        annotation = annotation[-1]["text"]

    text = self.processor.apply_chat_template(
      prompt_messages, tokenize=False,
      add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info_qwen(messages)
    if image_inputs:
      image_inputs = image_inputs[:self.max_images]
    if video_inputs:
      video_inputs = video_inputs[:self.max_images]

    mm_data = {}
    if image_inputs:
      mm_data["image"] = image_inputs
    if video_inputs:
      mm_data["video"] = video_inputs

    return {
      "vllm_inputs": {
        "prompt": text,
        "multi_modal_data": mm_data,
      },
      "annotation": annotation,
      "source": sample["json"]["source"],
      "__key__": sample["__key__"],
      "__url__": sample["__url__"],
      "metadata": sample["raw"].loc["metadata"],
    }

class Qwen3Dataset(DistributedDataset):
  def __init__(self,
               sources: Union[str, List[str]],
               rank: int = 0,
               world_size: int = 1,
               num_workers: int=8,
               seed: int=1024,
               system_prompt: str = "default",
               **kwargs):
    super().__init__(
      sources=sources, rank=rank, world_size=world_size,
      num_workers=num_workers, seed=seed, **kwargs)
    prompt_loader = PromptLoader()
    self.system_prompt = prompt_loader.load(system_prompt)
    self.tokenizer_path = kwargs.get("tokenizer_path", None)
    self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)

  def process(self,
              sample: Dict[str, Any]):
    assert "messages" in sample["json"], \
        f"sample must contain `messages` key, but got {sample['json'].keys()}"

    messages = sample["json"]["messages"]

    transformed_messages = []
    valid_message = True
    for msg in messages:
        role = msg.get("role")
        content_list = msg.get("content")
        if role and isinstance(content_list, list) and content_list and content_list[0].get("type") == "text":
            text_content = content_list[0].get("text")
            if text_content is not None:
                transformed_messages.append({"role": role, "content": text_content})
            else:
                valid_message = False
                break
        else:
            valid_message = False
            break
    
    if not valid_message or not transformed_messages:
      print(f"Warning: Message format is incorrect. Data: {messages}")

    if not messages or messages[0]["role"] != "system":
      system = {"role": "system", "content": self.system_prompt}
      messages.insert(0, system)

    text = self.tokenizer.apply_chat_template(
      transformed_messages,
      tokenize=False,
      add_generation_prompt=True
    )

    return {
      "vllm_inputs": {
        "prompt": text,
      },
      "annotation": "",
      "image_path": "",
      "video_path": "",
      "metadata": sample["raw"].loc["metadata"],
      "source": sample["json"]["source"],
      "__key__": sample["__key__"],
      "__url__": sample["__url__"],
    }
        
class KeyeDataset(DistributedDataset):
  "For vllm batch inference"
  def __init__(self,
               sources: Union[str, List[str]],
               rank: int = 0,
               world_size: int = 1,
               num_workers: int = 8,
               seed: int = 1024,
               system_prompt: str = "default",
               tokenizer_path: Optional[str] = None,
               **kwargs):
    super().__init__(
      sources=sources, rank=rank, world_size=world_size,
      num_workers=num_workers, seed=seed, **kwargs)
    prompt_loader = PromptLoader()
    self.system_prompt = prompt_loader.load(system_prompt)
    self.tokenizer_path = tokenizer_path
    self.min_pixels = \
      kwargs.get("min_pixels", 4 * 28 * 28)
    self.max_pixels = \
      kwargs.get("max_pixels", 1280 * 28 * 28)
    self.video_min_pixels = \
      kwargs.get("video_min_pixels", 4 * 28 * 28)
    self.video_max_pixels = \
      kwargs.get("video_max_pixels", 768 * 28 * 28)
    self.max_images = kwargs.get("max_images", 10)
    self.max_videos = kwargs.get("max_videos", 10)
    self.fps = kwargs.get("fps", 1.0)
    self.min_frames = kwargs.get("min_frames", 1)
    self.max_frames = kwargs.get("max_frames", 120)
    self.processor = AutoProcessor.from_pretrained(tokenizer_path, trust_remote_code=True)

  def fill_image_block(self,
                       block: Dict[str, Any],
                       sample: Optional[Dict[str, Any]] = None,
                       is_video: bool = False):

    assert block["image"] in sample,\
      f"image key={block['image']} not in sample., {sample.keys()}"
    if block["image"] in sample:
      block["image"] = sample[block["image"]]
    else:
      if isinstance(block["image"], str):
        block["image"] = load_image(block["image"])
  
    if block["image"].mode != "RGB":
      block["image"] = block["image"].convert("RGB")

    if is_video:
      block["min_pixels"] = self.video_min_pixels
      block["max_pixels"] = self.video_max_pixels
    else:
      block["min_pixels"] = self.min_pixels
      block["max_pixels"] = self.max_pixels

    return block

  def fill_video_block(self,
                       block: Dict[str, Any],
                       sample: Dict[str, Any]):

    if isinstance(block["video"], list):
      # fake image block list，复用fill_image_block
      if all([isinstance(image_block, str) for image_block in block["video"]]):
        block["video"] = [
          {"type": "image", "image": image_str} for image_str in block["video"]]
      for image in block["video"]:
        self.fill_image_block(image, sample, is_video=True)

    elif isinstance(block["video"], str) or isinstance(block["video"], bytes):
      # video in local tar, replace by video bytes
      if isinstance(block["video"], str) and block["video"] in sample:
        block["video"] = sample[block["video"]]
      
      if isinstance(block["video"], str) and not os.path.exists(block["video"]):
        raise ValueError(f"video file not exists: {block['video']}")
      # image list格式不提供默认fps，如果用户指定，会按照真实fps来计算timestamps
      if self.fps > 0:
        block["fps"] = self.fps
    else:
      raise ValueError(
        f"Unsupport video type. {type(block['video'])=}")

    # fill other params
    block["min_pixels"] = self.video_min_pixels
    block["max_pixels"] = self.video_max_pixels
    # video split params
    if self.min_frames > 0:
      block["min_frames"] = self.min_frames
    if self.max_frames > 0:
      block["max_frames"] = self.max_frames
    return block

  def process(self,
              sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    assert "messages" in sample["json"], \
        f"sample must contain `messages` key, but got {sample['json'].keys()}"

    messages = sample["json"]["messages"]
    prompt_messages = []
    for turn in messages:
      content = turn["content"]
      if isinstance(content, str):
        continue
      for block in content:
        if block["type"] == "image":
          self.fill_image_block(block, sample)
        elif block["type"] == "video":
          self.fill_video_block(block, sample)
        elif block["type"] == "text":
          continue
        else:
          raise ValueError(
            f"sample process error, unsupport value type: {block['type']}")

    if messages[0]["role"] != "system":
      system = {"role": "system", "content": self.system_prompt}
      messages.insert(0, system)

    for turn in messages:
      if turn["role"] == "assistant":
        break
      prompt_messages.append(turn)

    annotation = None
    if messages[-1]["role"] == "assistant":
      annotation = messages[-1]["content"]
      if isinstance(annotation, list):
        annotation = annotation[-1]["text"]

    text = self.processor.apply_chat_template(
      prompt_messages, tokenize=False,
      add_generation_prompt=True
    )

    # Use the capped prompt_messages to extract vision info so that
    # the number of placeholders matches the actual inputs passed to vLLM
    image_inputs, video_inputs, processor_kwargs = process_vision_info_keye(prompt_messages)

    if image_inputs:
      image_inputs = image_inputs[:self.max_images]
    if video_inputs:
      video_inputs = video_inputs[:self.max_videos]

    mm_data = {}
    if image_inputs:
      mm_data["image"] = image_inputs
    if video_inputs:
      mm_data["video"] = video_inputs
    return {
      "vllm_inputs": {
        "prompt": text,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": processor_kwargs
      },
      "annotation": annotation,
      "source": sample["json"]["source"],
      "__key__": sample["__key__"],
      "__url__": sample["__url__"],
      "metadata": sample["raw"].loc["metadata"],
    }


class OpenAIDataset(DistributedDataset):
  "For OAI batch inference"
  def __init__(self,
               sources: Union[str, List[str]],
               rank: int = 0,
               world_size: int = 1,
               num_workers: int = 8,
               seed: int = 1024,
               system_prompt: str = "default",
               tokenizer_path: Optional[str] = None,
               **kwargs):
    super().__init__(
      sources=sources, rank=rank, world_size=world_size,
      num_workers=num_workers, seed=seed, preload=False,\
        **kwargs)
    prompt_loader = PromptLoader()
    self.system_prompt = prompt_loader.load(system_prompt)
    self.tokenizer_path = tokenizer_path
    self.min_pixels = \
      kwargs.get("min_pixels", 4 * 28 * 28)
    self.max_pixels = \
      kwargs.get("max_pixels", 1280 * 28 * 28)
    self.video_min_pixels = \
      kwargs.get("video_min_pixels", 4 * 28 * 28)
    self.video_max_pixels = \
      kwargs.get("video_max_pixels", 768 * 28 * 28)
    self.max_images = kwargs.get("max_images", 10)
    self.max_videos = kwargs.get("max_videos", 10)
    self.fps = kwargs.get("fps", 1.0)
    self.min_frames = kwargs.get("min_frames", 1)
    self.max_frames = kwargs.get("max_frames", 120)

  def fill_image_block(self,
                       block: Dict[str, Any],
                       sample: Optional[Dict[str, Any]] = None,
                       is_video: bool = False):

    assert block["image"] in sample,\
      f"image key={block['image']} not in sample., {sample.keys()}"
  
    image = block.pop("image")
    if image in sample:
      image = sample[image]
    if not os.path.exists(image):
      image = f"data:image/jpeg;base64,{image}"

    block["type"] = "image_url"
    block["image_url"] = {
      "url": image
    }

    if is_video:
      block["min_pixels"] = self.video_min_pixels
      block["max_pixels"] = self.video_max_pixels
    else:
      block["min_pixels"] = self.min_pixels
      block["max_pixels"] = self.max_pixels

    return block

  def fill_video_block(self,
                       block: Dict[str, Any],
                       sample: Dict[str, Any]):
    assert isinstance(block["video"], str), f"Only mp4 format is supported."
    assert block["video"] in sample,\
      f"video key={block['video']} not in sample., {sample.keys()}"
  
    video = block.pop("video")
    if video in sample:
      video = sample[video]
    assert os.path.exists(video), f"Video path={video} doesn't exists."

    block["type"] = "video_url"
    block["video_url"] = {"url": video}

    block["min_pixels"] = self.video_min_pixels
    block["max_pixels"] = self.video_max_pixels

    return block

  def process(self,
              sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    assert "messages" in sample["json"], \
        f"sample must contain `messages` key, but got {sample['json'].keys()}"

    messages = sample["json"]["messages"]
    prompt_messages = []
    for turn in messages:
      content = turn["content"]
      if not isinstance(content, list):
        continue
      for block in content:
        if block["type"] == "image":
          self.fill_image_block(block, sample)
        elif block["type"] == "video":
          self.fill_video_block(block, sample)
        elif block["type"] == "text":
          continue
        # TODO: support image_url type
        else:
          raise ValueError(
            f"sample process error, unsupport value type: {block['type']}")
    # TODO: 默认会添加一个system_prompt，后续支持不添加sys
    if messages[0]["role"] != "system":
      system = {"role": "system", "content": self.system_prompt}
      messages.insert(0, system)

    for turn in messages:
      if turn["role"] == "assistant":
        break
      prompt_messages.append(turn)

    annotation = None
    if messages[-1]["role"] == "assistant":
      annotation = messages[-1]["content"]
      if isinstance(annotation, list):
        annotation = annotation[-1]["text"]

    return {
      "messages": prompt_messages,
      "annotation": annotation,
      "source": sample["json"]["source"],
      "__key__": sample["__key__"],
      "__url__": sample["__url__"],
      "metadata": sample["raw"].loc["metadata"],
    }
