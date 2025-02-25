"""Batch inference for Qwen2-VL"""
from absl import flags, app
import json
import collections
import os
import sys
import logging
import torch
import pandas as pd
from typing import Dict, List

# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, project_root)

from tqdm import tqdm
from vllm import LLM, SamplingParams
from wenjuan_infer_dataset import WenJuanInferDataset
from torch.utils.data import DataLoader


FLAGS = flags.FLAGS

flags.DEFINE_string(
  "model_name_or_path", "/llm_reco_ssd/huqigen/recovlm/recovlm/benchmark/vlm_model_infer_hqg", "The path or name of model."
)

flags.DEFINE_string(
  "parquet_path", "/llm_reco_ssd/huqigen/dataset/wenjuan_sft/photo_0210_11w_cot_v2/photo_0210_11w_sft_data-test.parquet", "The path or name of model."
)

# flags.DEFINE_string(
#   "parquet_path", "viewfs://hadoop-lt-cluster/home/reco_wl/mpi/huqigen/recovlm_dataset/wenjuan_sft/0210_11w/photo_0210_11w_sft_data-test.parquet", "The path or name of model."
# )

flags.DEFINE_float(
  "top_p", 0.8, "The top_p params"
)

flags.DEFINE_float(
  "temperature", 0.6, "The temperature params."
)

flags.DEFINE_integer(
  "max_tokens", 2048, "The max tokens to generate."
)

flags.DEFINE_integer(
  "tp", 4, "The tensor_parallel_size"
)

flags.DEFINE_integer(
  "votes", 1, "The number of candidates in majority voting."
)

flags.DEFINE_string(
  "system_prompt", None, "The system prompt to use."
)

flags.DEFINE_string(
  "input_path", None, "The parquet file path for test data."
)

flags.DEFINE_integer(
  "max_samples", 1000, "The maximum num of samples to inference."
)

flags.DEFINE_string(
  "output_path", "wenjuan_response.jsonl", "The path of file to write results." 
)

flags.DEFINE_integer(
  "limit_mm_per_prompt", 3, "The maximum images of mm_input per prompt"
)

flags.DEFINE_integer(
  "num_images", 20, "The number of images of per instance."
)

flags.DEFINE_integer(
  "max_text_len", 1000, "The max text length per field."
)

flags.DEFINE_integer(
  "batch_size", 32, "The batch size for inference."
)

flags.DEFINE_float(
  "repetition_penalty", 1.05, "The maximum images of mm_input per prompt"
)

flags.DEFINE_string(
  "hdfs_user", "mpi", "The HDFS user name when reading from HDFS."
)

flags.DEFINE_integer(
  "num_samples", None, "Number of samples to infer. If None, process all samples."
)

flags.DEFINE_integer(
  "max_frames", 32, "The maximum number of frames in a video."
)

flags.DEFINE_string(
  "columns", None, "The columns to include in the dataset."
)

flags.DEFINE_string(
  "user", "mpi", "The HDFS user name when reading from HDFS."
)

flags.DEFINE_integer(
  "limit", 20000, "The maximum number of samples to read from the dataset."
)

flags.DEFINE_string("metrics_output_file", "infer_metric.txt", "Path to the file to output final metrics (accuracy etc.)")

def collate_fn(samples):
  batch = collections.defaultdict(list)
  for sample in samples:
    for key, item in sample.items():
      batch[key].append(item)
  return batch

def extract_satisfaction(response: str) -> str:
    """从响应中提取满意度结果"""
    try:
        # 尝试从标准格式中提取
        if "用户对视频的满意度结果是：" in response:
            result = response.split("用户对视频的满意度结果是：")[-1].strip()
            if result in ["满意", "不满意"]:
                return result
        
        # 尝试从 ### 分隔符格式中提取
        if "###" in response:
            conclusion = response.split("###")[-1].strip()
            if conclusion in ["满意", "不满意"]:
                return conclusion
            
        if "不满意" in response:
            return "不满意"
        elif "满意" in response:
            return "满意"
                
        return None
    except Exception as e:
        print(f"Error extracting satisfaction: {e}")
        return None

def parse_prompt(prompt):
    """解析prompt内容，提取assistant回复
    
    Args:
        prompt (str): 原始prompt文本
        
    Returns:
        str or None: assistant的回复内容
    """
    try:
        # 打印原始prompt内容，用于调试
        print("\n=== Original Prompt Content ===")
        print(prompt)
        print("=============================\n")
        
        # 按照对话格式分割
        parts = prompt.split("<|im_start|>")
        for part in parts:
            if "assistant" in part:
                # 提取assistant部分的内容
                content = part.split("<|im_end|>")[0].strip()
                content = content.replace("assistant\n", "").strip()
                return content
    except Exception as e:
        print(f"Error parsing prompt: {e}")
        print(f"Prompt content: {prompt}")
    return None

def calculate_accuracy(true_labels: list, pred_labels: list) -> float:
    """计算准确率"""
    if len(true_labels) != len(pred_labels) or len(true_labels) == 0:
        return 0.0
    correct = sum(1 for t, p in zip(true_labels, pred_labels) if t == p)
    return correct / len(true_labels)

def extract_photo_id_from_messages(messages: List[dict]) -> str:
    """从消息中的图片名称提取 photo_id"""
    try:
        for msg in messages:
            if msg.get('role') == 'user':
                content = msg.get('content', [])
                if isinstance(content, list):
                    for item in content:
                        if item.get('type') == 'video':
                            video = item.get('video', [])
                            if video and isinstance(video, list):
                                first_frame = video[0]
                                if first_frame.get('type') == 'image':
                                    image_name = first_frame.get('image', '')
                                    if '_' in image_name:
                                        return image_name.split('_')[0]
        return None
    except Exception as e:
        print(f"Error extracting photo_id from messages: {e}")
        return None

def load_true_labels(parquet_path: str) -> Dict[str, str]:
    """从parquet文件中加载真实标签"""
    try:
        print(f"\nLoading true labels from: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        print(f"Total rows: {len(df)}")
        
        true_labels = {}
        error_count = 0
        
        for idx, row in df.iterrows():
            try:
                messages = row.get('messages', [])
                
                if isinstance(messages, str):
                    try:
                        messages = json.loads(messages)
                    except Exception as e:
                        error_count += 1
                        continue
                
                if not isinstance(messages, list):
                    error_count += 1
                    continue
                
                # 从消息中提取 photo_id
                photo_id = extract_photo_id_from_messages(messages)
                if not photo_id:
                    error_count += 1
                    continue
                
                # 提取满意度结果
                for msg in messages:
                    if msg.get('role') == 'assistant':
                        content = msg.get('content', '')
                        label = extract_satisfaction(content)
                        if label:
                            true_labels[photo_id] = label
                            break
                
            except Exception as e:
                error_count += 1
                continue
        
        print(f"\nLabel extraction summary:")
        print(f"Total rows processed: {len(df)}")
        print(f"Labels extracted: {len(true_labels)}")
        print(f"Errors encountered: {error_count}")
        return true_labels
        
    except Exception as e:
        print(f"Error loading parquet file: {e}")
        return {}

def main(_):
  # Pass the default decoding hyperparameters of Qwen2.5-7B-Instruct
  # max_tokens is for the maximum length for generation.
  sampling_params = SamplingParams(
    temperature=FLAGS.temperature, top_p=FLAGS.top_p,
    repetition_penalty=FLAGS.repetition_penalty, max_tokens=FLAGS.max_tokens)
  # Input the model name or path. Can be GPTQ or AWQ models.
  llm = LLM(
    model=FLAGS.model_name_or_path,
    tensor_parallel_size=FLAGS.tp,
    limit_mm_per_prompt={
      "image": FLAGS.limit_mm_per_prompt,
      "video": FLAGS.limit_mm_per_prompt
    }
  )

  dataset = WenJuanInferDataset(
    parquet_path=FLAGS.parquet_path,
    system_prompt=FLAGS.system_prompt,
    model_name_or_path=FLAGS.model_name_or_path,
    max_text_len=FLAGS.max_text_len,
    max_frames=FLAGS.max_frames,
    columns=FLAGS.columns,
    user=FLAGS.user,
    limit=FLAGS.limit
  )

  processed_samples = 0
  true_labels = {}
  
  # 加载真实标签
  true_labels = load_true_labels(FLAGS.parquet_path)
  
  # 初始化列表用于存储标签
  true_labels_list = []
  pred_labels_list = []
  
  with open(FLAGS.output_path, "w", encoding="utf-8") as f:
    for batch in tqdm(DataLoader(dataset,
                               batch_size=FLAGS.batch_size,
                               collate_fn=collate_fn)):
      print("\n=== Debug: Batch structure ===")
      print(f"Batch keys: {batch.keys()}")
      print(f"Batch size: {len(batch['inputs'])}")
      
      batch_rsp = [[] for _ in range(len(batch["inputs"]))]
      for _ in range(FLAGS.votes):
        outputs = llm.generate(batch["inputs"], sampling_params)
        for idx, output in enumerate(outputs):
          batch_rsp[idx].append(output.outputs[0].text)
      
      for idx, rsp in enumerate(batch_rsp):
        photo_id = batch["photo_id"][idx]
        
        # 获取真实标签
        true_label = true_labels.get(photo_id)
        
        # 获取预测标签
        pred_label = None
        for response in rsp:
            pred_label = extract_satisfaction(response)
            if pred_label:
                break
        
        if true_label and pred_label:
            true_labels_list.append(true_label)
            pred_labels_list.append(pred_label)
        else:
            logging.warning(f"Sample {processed_samples + 1}: Failed to extract labels. true={true_label}, pred={pred_label}")
        
        # 保存结果
        f.write(json.dumps({
            "req_id": batch["req_id"][idx],
            "photo_id": photo_id,
            "prompt": batch["inputs"][idx]["prompt"],
            "rsp": rsp,
            "true_label": true_label,
            "pred_label": pred_label
        }, ensure_ascii=False) + "\n")
        
        processed_samples += 1
        if FLAGS.num_samples is not None and processed_samples >= FLAGS.num_samples:
            logging.info(f"\nReached sample limit ({FLAGS.num_samples}), stopping...")
            break
  
    # 计算并打印指标
    if true_labels_list and pred_labels_list:
        accuracy = calculate_accuracy(true_labels_list, pred_labels_list)
        metrics_info = {
            "accuracy": accuracy,
            "total_samples": len(true_labels_list),
            "correct_predictions": int(accuracy * len(true_labels_list))
        }
        print(f"\nAccuracy: {accuracy:.4f}")
        print(f"Total samples evaluated: {len(true_labels_list)}")
        print(f"Correct predictions: {metrics_info['correct_predictions']}")
        
        # 如果配置了输出指标文件，则写入该文件
        if FLAGS.metrics_output_file:
            with open(FLAGS.metrics_output_file, "w", encoding="utf-8") as mf:
                mf.write(json.dumps(metrics_info, ensure_ascii=False, indent=2))
    else:
        logging.warning("No valid labels extracted, skipping metrics calculation")

if __name__ == "__main__":
  app.run(main)