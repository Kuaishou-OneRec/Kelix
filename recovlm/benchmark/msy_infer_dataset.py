"""I2I Pairwise Dataset"""
import re
import numpy as np
import collections
import json
import os
import sys
import traceback
import base64
from io import BytesIO
from PIL import Image
import uuid
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from recovlm.models.qwen3siglip.processing_qwen3siglip import Qwen3SiglipProcessor_siglip
from torch.utils.data import DataLoader
from qwen_vl_utils import process_vision_info

# 使用正确的相对导入路径
from dataset import ParquetDataset
from loader import PromptLoader

def is_null(text):
  if not text:
    return True
  if isinstance(text, float) and np.isnan(text):
    return True
  if text == "null":
    return True
  if text == "该视频暂时没有评论":
    return True
  return False

def format_text(doc, max_text_len=1000):
  items = []
  for key, text in doc.items():
    if not is_null(text):
      items.append(f"{key}: {str(text)[:max_text_len]}")
  return "\n".join(items)

class MsyInferDataset(ParquetDataset):
  """I2I Pairwise Relevance"""
  def __init__(self, 
               parquet_path,  # 明确声明必需的参数
               system_prompt=None,
               model_name_or_path=None,
               max_text_len=512,
               max_frames=32,
               columns=None,
               user="mpi",
               limit=None,
               enable_remove_comment=False,
               **kwargs):
    # 初始化父类
    super().__init__(path=parquet_path, columns=columns, user=user, limit=limit)
    
    # 初始化 processor
    if model_name_or_path:
      try:
        self.processor = Qwen3SiglipProcessor_siglip.from_pretrained(model_name_or_path)
      except Exception as e:
        print(f"Error loading processor: {e}")
        print("Using default chat template")
        self.processor = None
    else:
      self.processor = None
      
    self.system_prompt = system_prompt or "You are a helpful assistant."
    self.model_name_or_path = model_name_or_path
    self.max_text_len = max_text_len
    self.max_frames = max_frames
    
    if parquet_path.endswith(".parquet"):
      self.total_rows = pq.read_metadata(parquet_path).num_rows
    elif parquet_path.endswith(".json"):
      with open(parquet_path, "r") as f:
        self.total_rows = len(json.load(f))
    else:
      self.total_rows = 0
    if limit is not None:
      self.total_rows = min(self.total_rows, limit)
    self.enable_remove_comment = enable_remove_comment

  def __iter__(self):
    """重写父类的__iter__方法，处理每个样本"""
    for item in super().__iter__():
      try:
        # 获取messages
        messages = item.get("messages", [])
        if isinstance(messages, str):
          try:
            messages = json.loads(messages)
          except Exception as e:
            print(f"Error parsing messages JSON: {e}")
            messages = [{"role": "user", "content": messages}]
        text = self.processor.apply_chat_template(
          messages,
          tokenize=False,
          add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        mm_data = {}
        if image_inputs is not None:
          mm_data["images"] = image_inputs
        if video_inputs is not None:
          mm_data["videos"] = video_inputs
        inputs  = self.processor(
          text=[text],
          **mm_data,
          padding=True,
          return_tensors="pt",
        )
        yield {
          "inputs": inputs
        }
        
      except Exception as e:
        print(f"Error processing item: {e}")
        yield {
          "inputs": None
        }

  def __len__(self):
    """返回数据集的总行数"""
    return self.total_rows

if __name__ == "__main__":
  dataset = MsyInferDataset(
    parquet_path="/llm_reco_ssd/luoxinchen/RecoVLM/Benchmark/dataset/MMBench/en/dev-00000-of-00001.parquet",
    model_name_or_path="/llm_reco_ssd/zhouyang12/models/Qwen3-1.7B-siglip",
    user='mpi'
  )
  for batch in DataLoader(dataset, batch_size=2, shuffle=False):
    for idx, item in enumerate(batch):
      print(idx, item)
    break