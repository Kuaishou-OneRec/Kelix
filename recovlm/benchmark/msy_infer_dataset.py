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
from recovlm.utils.qwen_vl_utils import process_vision_info
import torch
# 使用正确的相对导入路径
from dataset import ParquetDataset
from loader import PromptLoader
import logging

def format_text(doc, max_text_len=1000):
  items = []
  for key, text in doc.items():
    if not is_null(text):
      items.append(f"{key}: {str(text)[:max_text_len]}")
  return "\n".join(items)



def parse_options_and_answer(options, answer):
    if all([len(option) == 1 and option.isalpha() for option in options]):
        options = [option.upper() for option in options]

    options_list = list()
    for idx, option in enumerate(options):
        ch = chr(ord('A') + idx)
        options_list.append("{}. {}".format(ch, option))
    answer = int(answer)
    assert answer in [0, 1, 2, 3], str(type(answer)) + "   " + str(answer)
    return "\n".join(options_list), chr(ord('A') + answer)

def AI2DTransform(sample) -> list:
  question = sample['question']
  answer = sample['answer']
  options = sample['options']
  image_bytes = sample['image']['bytes']
  options_str, answer = parse_options_and_answer(options, answer)
  messages = [
    {
      "role": "user",
      "content": [
        {
          "type": "image",
          "image": Image.open(BytesIO(image_bytes))
        },
        {
          "type": "text",
          "text": question
        },
        {
          "type": "text",
          "text": options_str
        },
        {
          "type": "text",
          "text": "\n Answer with the letter."
        }
      ]
    },
    {
      "role": "assistant",
      "content": [
        {
          "type": "text",
          "text": answer
        }
      ]
    }
  ]
  return messages
  


def Benchmark_v21Transform(sample) -> list:
  sample = sample['annotations']
  # Handle both string and dictionary annotations
  if isinstance(sample, str):
    sample = json.loads(sample)
  question = sample['question']
  answer = sample['answer']
  image_path = sample['image_path']
  image = Image.open(image_path)
  messages = [
    {
      "role": "user",
      "content": [
        {
          "type": "image",
          "image": image
        },
        {
          "type": "text",
          "text": question
        }
      ]
    },
    {
      "role": "assistant",
      "content": [
        {
          "type": "text",
          "text": answer
        }
      ]
    }
  ]
  return messages
  
  

def MathVistaTransform(sample) -> list:
  sample = sample['annotations']
  # Handle both string and dictionary annotations
  if isinstance(sample, str):
    sample = json.loads(sample)
  question = sample['question']
  answer = sample['answer']
  image_path = sample['image_path']
  image = Image.open(image_path)
  messages = [
    {
      "role": "user",
      "content": [
        {
          "type": "image",
          "image": image
        },
        {
          "type": "text",
          "text": question
        }
      ]
    },
    {
      "role": "assistant",
      "content": [
        {
          "type": "text",
          "text": answer
        }
      ]
    }
  ]
  return messages
  

def mmstarTransform(sample) -> list:
  image = sample['image']
  question = sample['question']
  answer = sample['answer']
  messages = [
    {
      "role": "user",
      "content": [
        {
          "type": "image",
          "image": Image.open(BytesIO(image))
        },
        {
          "type": "text",
          "text": question
        }
      ]
    },
    {
      "role": "assistant",
      "content": [
        {
          "type": "text",
          "text": answer
        }
      ]
    }
  ]
  return messages

def MMTBenchTransform(sample) -> list:
  sample = sample['annotations']
  # Handle both string and dictionary annotations
  if isinstance(sample, str):
    sample = json.loads(sample)
  question = sample['question']
  answer = sample['answer']
  image_path = sample['image_path']
  image = Image.open(image_path)
  messages = [
    {
      "role": "user",
      "content": [
        {
          "type": "image",
          "image": image
        },
        {
          "type": "text",
          "text": question
        }
      ]
    },
    {
      "role": "assistant",
      "content": [
        {
          "type": "text",
          "text": answer
        }
      ]
    }
  ]
  return messages
  


def MMETransform(sample) -> list:
  sample = sample['annotations']
  # Handle both string and dictionary annotations
  if isinstance(sample, str):
    sample = json.loads(sample)
  question = sample['question']
  answer = sample['answer']
  image_path = sample['image_path']
  image = Image.open(image_path)
  messages = [
    {
      "role": "user",
      "content": [
        {
          "type": "image",
          "image": image
        },
        {
          "type": "text",
          "text": question
        }
      ]
    },
    {
      "role": "assistant",
      "content": [
        {
          "type": "text",
          "text": answer
        }
      ]
    }
  ]
  return messages

def MMBenchTransform(sample) -> list:
  index = sample['index']
  image = sample['image']['bytes']
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
                  "image": Image.open(BytesIO(image))
              },
              {
                  "type": "text", 
                  "text": prompt
              },
          ]

      },
      {
        "role": "assistant",
        "content": [
          {
            "type": "text",
            "text": answer
          }
        ]
      }
  ] 
  return messages

def OCRBenchTransform(sample) -> list:
  question = sample['question'] 
  image = sample['image']['bytes']
  answer = sample['answer']
  messages = [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": Image.open(BytesIO(image))},
        {"type": "text", "text": question}
      ],
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": answer}
      ]
    }
  ]
  return messages

transform_func_map = {
  "MMBench": MMBenchTransform,
  "OCRBench": OCRBenchTransform,
  "MMBenchCn": MMBenchTransform,
  "MME": MMETransform,
  "MMTBench": MMTBenchTransform,
  "MMStar": mmstarTransform,
  "MathVista": MathVistaTransform,
  "Benchmark_v21": Benchmark_v21Transform,
  "AI2D": AI2DTransform
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
    self.start_id = 151644 
    self.end_id = 151645#use to locate the answer in the input_ids
    
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
      try:
        messages = self.transform_func(item)
        print(messages,'msylalalallla')
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
        inputs = self.processor(
          text=[text],
          **mm_data,
          padding=True,
          return_tensors="pt",
        )
        input_ids = inputs["input_ids"]
        
        answer_idx_list = []
        # 将tensor转换为list以便使用index方法
        input_ids_list = input_ids[0].tolist()
        try:
            # 找到所有start_id的位置
            start_positions = [i for i, x in enumerate(input_ids_list) if x == self.start_id]
            # 找到所有end_id的位置
            end_positions = [i for i, x in enumerate(input_ids_list) if x == self.end_id]
            
            # 确保有足够的start和end标记
            if len(start_positions) >= 3 and len(end_positions) >= 3:
                # 获取第三对的位置
                start_pos = start_positions[2]  # 第三个start_id的位置
                end_pos = end_positions[2]      # 第三个end_id的位置
                
                # 验证end_pos在start_pos之后
                if end_pos > start_pos:
                    answer_idx_list.append((start_pos+1, end_pos-1))
                    # print("Found third pair positions:", answer_idx_list)
                    # print("Input IDs:", input_ids_list)
                    # print("output IDs:", input_ids_list[start_pos:end_pos])
                else:
                    logging.warning("Found end_id before start_id, skipping")
            else:
                logging.warning(f"Not enough start/end tokens found. Found {len(start_positions)} start tokens and {len(end_positions)} end tokens")
                continue
                
        except Exception as e:
            logging.warning(f"Error finding token positions: {e}")
            continue
            
        if not answer_idx_list:
            # 如果没有找到任何有效的起始位置，跳过这个样本
            logging.warning("No valid start positions found, skipping sample")
            continue
            
        yield {
          "inputs": inputs,
          "answer_idx_list": answer_idx_list
        }
        
      except Exception as e:
        logging.error(f"Error processing item: {e}")
        continue

  def __len__(self):
    """返回数据集的总行数"""
    return self.total_rows

if __name__ == "__main__":
  # dataset = MsyInferDataset(
  #   dataset_name="MMBench",
  #   parquet_path="/llm_reco_ssd/luoxinchen/RecoVLM/Benchmark/dataset/MMBench/en/dev-00000-of-00001.parquet",
  #   model_name_or_path="/llm_reco_ssd/zhouyang12/models/Qwen3-1.7B-siglip",
  #   user='mpi'
  # )
  dataset = MsyInferDataset(
    dataset_name="OCRBench",
    parquet_path="/llm_reco_ssd/luoxinchen/RecoVLM/Benchmark/dataset/OCRBench/data/test-00000-of-00001.parquet",
    model_name_or_path="/llm_reco_ssd/zhouyang12/models/Qwen3-1.7B-siglip",
    user='mpi'
  )

  for batch in DataLoader(dataset, batch_size=1, shuffle=False):
    print("Batch type:", type(batch))
    print("Batch contents:", batch)
    for idx, item in enumerate(batch):
      print("Item type:", type(item))
      print("Item contents:", item)
      if isinstance(item, dict):
        print("Inputs:", item.get('inputs'))
      else:
        print("Item is not a dictionary:", item)
    break