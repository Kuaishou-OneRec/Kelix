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

from torch.utils.data import DataLoader
from transformers import AutoProcessor
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

class WenJuanInferDataset(ParquetDataset):
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
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
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
    
    # 读取数据以获取长度
    if parquet_path.startswith("viewfs") or parquet_path.startswith("hdfs://"):
      self.fs = pa.hdfs.connect(user=user)
      with self.fs.open(parquet_path, 'rb') as f:
        self.total_rows = pq.read_metadata(f).num_rows
    else:
      self.total_rows = pq.read_metadata(parquet_path).num_rows
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
        
        # 获取photo_id
        photo_id = self._extract_photo_id(item.get("images", {}))
        if not photo_id:
          photo_id = str(uuid.uuid1())
        
        # 处理消息内容，传入当前item
        filtered_messages = self._process_messages(messages, photo_id, item)
        
        # 格式化prompt
        infer_prompt = self._format_prompt(filtered_messages)
        
        # 处理多模态数据
        image_inputs, video_inputs = process_vision_info(filtered_messages)
        mm_data = {}
        if image_inputs is not None:
          mm_data["images"] = image_inputs
        if video_inputs is not None:
          mm_data["videos"] = video_inputs
        
        yield {
          "req_id": str(uuid.uuid1()),
          "photo_id": photo_id,
          "inputs": {
            "prompt": infer_prompt,
            **mm_data
          },
          "messages": messages
        }
        
      except Exception as e:
        print(f"Error processing item: {e}")
        yield {
          "req_id": str(uuid.uuid1()),
          "photo_id": "",
          "inputs": {
            "prompt": "",
            "images": [],
            "videos": []
          },
          "messages": []
        }

  def __len__(self):
    """返回数据集的总行数"""
    return self.total_rows

  def _format_prompt(self, messages):
    """格式化对话内容为prompt"""
    try:
      if self.processor:
        # 使用模型的chat template
        return self.processor.apply_chat_template(
          messages,
          tokenize=False,
          add_generation_prompt=True
        )
      else:
        # 使用默认的格式化方法
        formatted_messages = []
        for msg in messages:
          role = msg.get("role", "")
          content = msg.get("content", "")
          formatted_messages.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        return "\n".join(formatted_messages)
    except Exception as e:
      print(f"Error formatting prompt: {e}")
      print(f"Messages: {messages}")
      # 返回一个基本的格式化结果
      return "\n".join([f"{msg.get('role', '')}: {msg.get('content', '')}" for msg in messages])

  def _remove_assistant_content(self, prompt):
    """移除 prompt 中的 assistant 内容"""
    try:
      parts = []
      current_part = []
      
      # 按照对话标记分割
      segments = prompt.split("<|im_start|>")
      for segment in segments:
        if not segment.strip():
          continue
          
        if "assistant" not in segment:
          # 保留非 assistant 部分
          current_part.append("<|im_start|>" + segment)
        
      return "".join(current_part)
    except Exception as e:
      print(f"Error removing assistant content: {e}")
      return prompt
  def _remove_comment(self, text):
    """移除文本中的评论相关内容"""
    # 移除整行评论内容
    text = re.sub(r"对应的短视频平台内用户评论内容是：.*?\n", "", text)
    return text

  def _get_image_data(self, photo_id, image_id, current_item):
    """从images字段获取实际的图片数据，并转换为PIL Image对象"""
    try:
      images = current_item.get('images', {})
      if isinstance(images, str):
        images = json.loads(images)
      
      image_key = f"{photo_id}_{image_id}"
      if image_key in images:
        # 获取base64数据
        image_data = images[image_key]
        if isinstance(image_data, dict) and 'image' in image_data:
          image_data = image_data['image']
        
        # 将base64转换为PIL Image对象
        if isinstance(image_data, str):
          try:
            # 如果base64字符串包含header，去掉header
            if ',' in image_data:
              image_data = image_data.split(',', 1)[1]
            # 解码base64并创建图片对象
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes))
            return image
          except Exception as e:
            print(f"Error converting base64 to image: {e}")
            return None
    except Exception as e:
      print(f"Error getting image data for {photo_id}_{image_id}: {e}")
    return None

  def _process_messages(self, messages, photo_id, current_item):
    """处理messages，移除assistant回复并添加新的提示"""
    filtered_messages = [
      msg for msg in messages 
      if msg.get("role") in ["system", "user"]
    ]
    
    # 限制文本长度
    for msg in filtered_messages:
      if isinstance(msg.get("content"), str):
        msg["content"] = msg["content"][:self.max_text_len]
      elif isinstance(msg.get("content"), list):
        for item in msg["content"]:
          if isinstance(item, dict) and item.get("type") == "text":
            item["text"] = item["text"][:self.max_text_len]
            if self.enable_remove_comment:
              item["text"] = self._remove_comment(item["text"])
    
    for msg in reversed(filtered_messages):
      if msg.get("role") == "user":
        if isinstance(msg["content"], list):
          # 处理视频内容
          for content in msg["content"]:
            if content.get("type") == "video":
              video_frames = []
              # 限制帧数
              for frame in list(content.get("video", []))[:self.max_frames]:
                if isinstance(frame, dict) and frame.get("type") == "image":
                  image_id = frame["image"].split("_")[-1]
                  image = self._get_image_data(photo_id, image_id, current_item)
                  if image:
                    # 调整图片大小
                    image = self._resize_image(image)
                    video_frames.append(image)
                    # 如果已经达到最大帧数，就停止
                    if len(video_frames) >= self.max_frames:
                      break
              content["video"] = video_frames
          
          msg["content"].append({
            "type": "text",
            "text": "\n请先详细分析，不要直接给出结论，然后根据你的分析，给出结论。照如下格式进行输出：将分析过程与最终的结论用`###`进行分隔，结论部分只需要输出\"满意\"或者\"不满意\"，不要输出其他多余信息。"
          })
        elif isinstance(msg["content"], str):
          msg["content"] += "\n请先详细分析，不要直接给出结论，然后根据你的分析，给出结论。照如下格式进行输出：将分析过程与最终的结论用`###`进行分隔，结论部分只需要输出\"满意\"或者\"不满意\"，不要输出其他多余信息。"
        break
        
    return filtered_messages

  def _resize_image(self, image):
    """调整图片大小以减少token数量"""
    target_size = 224  # 使用更小的目标尺寸
    w, h = image.size
    if w > h:
      new_w = target_size
      new_h = int(h * target_size / w)
    else:
      new_h = target_size
      new_w = int(w * target_size / h)
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)

  def _truncate_text(self, text, max_len=1000):
    """截断文本到指定长度"""
    if text and len(text) > max_len:
      return text[:max_len]
    return text

  def _extract_photo_id(self, images_str):
    """从images字段中提取photo_id"""
    try:
      images = json.loads(images_str) if isinstance(images_str, str) else images_str
      # images是一个字典，key的格式是"photo_id_idx"
      if images and len(images) > 0:
        # 获取第一个key并提取photo_id
        first_key = next(iter(images.keys()))
        photo_id = first_key.split('_')[0]
        return photo_id
    except Exception as e:
      print(f"Error extracting photo_id from images: {str(e)}")
    return None

if __name__ == "__main__":
  dataset = WenJuanInferDataset(
    parquet_path="viewfs://xxx/path/to/test.parquet",
    model_name_or_path="/llm_reco_ssd/zhouyang12/models/Qwen2-VL-72B-Instruct",
    user='mpi'
  )
  for batch in DataLoader(dataset, batch_size=2, shuffle=False):
    for idx, item in enumerate(batch):
      print(idx, item)
    break