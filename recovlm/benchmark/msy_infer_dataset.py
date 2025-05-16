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
rom recovlm.utils.qwen_vl_utils import process_vision_info

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

def image_to_PIL(image):
  #images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`)
  #convert image bytes to PIL.Image.Image
  if isinstance(image, bytes):
    image = Image.open(BytesIO(image))
  return image

def MMBenchTransform(sample) -> list:
  index = sample['index']
  image = sample['image']
  answer = sample['answer']
  hint = sample['hint'] if sample['hint'] else 'N/A'
  question = sample['question']
  multiple_choices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

  # choices = sample['choices']
  # choice_list = []
  # for i, c in enumerate(choices):
  #     choice_list.append('{}. {}'.format(multiple_choices[i], c))
  # choice_txt = '\n'.join(choice_list)
  choice_list = []
  for i,c in enumerate(multiple_choices):
    #if sample got options column, use it
    if c in sample:
      choice_list.append('{}. {}'.format(c, sample[c]))
  choice_txt = '\n'.join(choice_list)

  prompt = "hint: {}\nquestion: {}\noptions: {}\nanswer:".format(hint, question, choice_txt)
  messages = [
      {
          "role": "user",
          "content": [
              {
                  "type": "image",
                  "image": image_to_PIL(image)
              },
              {
                  "type": "text", 
                  "text": prompt
              },
          ]

      },
      {
        "role": "assistant",
        "content": answer
      }
  ] 
  return messages

def OCRBenchTransform(sample) -> list:
  question = sample['question'] 
  image = sample['image']
  answer = sample['answer']
  messages = [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": image_to_PIL(image)},
        {"type": "text", "text": question}
      ],
    },
    {
      "role": "assistant",
      "content": answer
    }
  ]
  return messages

transform_func_map = {
  "MMBench": MMBenchTransform,
  "OCRBench": OCRBenchTransform
}


class MsyInferDataset(ParquetDataset):
  """I2I Pairwise Relevance"""
  def __init__(self, 
               dataset_name,
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
    try:
      self.transform_func = transform_func_map[dataset_name]
    except KeyError:
      raise ValueError(f"Dataset name {dataset_name} not found in transform_func_map")

  def __iter__(self):
    """重写父类的__iter__方法，处理每个样本"""
    for item in super().__iter__():
      # try:
      messages = self.transform_func(item)
      print(messages)
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
        
      # except Exception as e:
      #   print(f"Error processing item: {e}")
      #   yield {
      #     "inputs": None
      #   }

  def __len__(self):
    """返回数据集的总行数"""
    return self.total_rows

if __name__ == "__main__":
  dataset = MsyInferDataset(
    dataset_name="MMBench",
    parquet_path="/llm_reco_ssd/luoxinchen/RecoVLM/Benchmark/dataset/MMBench/en/dev-00000-of-00001.parquet",
    model_name_or_path="/llm_reco_ssd/zhouyang12/models/Qwen3-1.7B-siglip",
    user='mpi'
  )
  for batch in DataLoader(dataset, batch_size=1, shuffle=False):
    for idx, item in enumerate(batch):
      print(idx, item)
    break