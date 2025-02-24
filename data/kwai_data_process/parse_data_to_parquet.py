"""
数据解析工具 (Data Parser Tool)

功能描述:
    将图片数据和文本标注转换为统一的Parquet格式，支持数据集自动分割和分布式存储。

主要特性:
    - 支持多数据源: Excel文件或HDFS
    - 支持多图片目录输入
    - 自动分割训练集(90%)和测试集(10%)
    - 支持多线程并行处理
    - 支持分片存储和HDFS上传
    - 自动生成数据集索引

配置参数 (config.yaml):
1. 必需参数:
    image_folder: str
        图片目录路径，多个路径用分号分隔
        示例: "/path1/images;/path2/images"
    
    data_source: str
        数据源类型: 'excel' 或 'hdfs'
    
    parquet_path: str
        Parquet文件输出路径
    
    txt_file_path: str
        CoT结果文件路径

2. 数据源相关参数 (根据data_source选择):
    excel_file_path: str
        Excel文件路径 (data_source='excel'时必需)
    
    hdfs_path: str
        HDFS数据路径 (data_source='hdfs'时必需)

3. 可选参数:
    num_shards: int
        训练集分片数量 (默认: 1)
        注意: 当数据量不足时，实际分片数可能小于此值
    
    hdfs_output_path: str
        HDFS输出路径 (默认: 空)
        如果配置，则自动上传到HDFS
    
    index_file: str
        索引文件路径 (默认: 'dataset_index.json')
    
    num_workers: int
        并行处理线程数 (默认: 1)
    
    source_name: str
        数据源标识 (默认: 'wenjuan_photo_0207_1k')
    
    batch_size: int
        批处理大小 (默认: 1000)
    
    resize_config:
        max_height: int (默认: 640)
        max_width: int (默认: 640)

输出文件:
    训练集:
        格式: {parquet_path}-train-{index}-of-{total}.parquet
        说明: 根据num_shards参数分片存储
    
    测试集:
        格式: {parquet_path}-test.parquet
        说明: 单个文件，不分片
    
    索引文件:
        格式: {index_file}
        内容: 训练集文件的HDFS路径列表

使用示例:
    python parse_data_to_parquet.py config.yaml

注意事项:
    1. 确保输入路径有效且具有访问权限
    2. 大数据集建议调整batch_size和num_workers
    3. HDFS操作需要配置好Hadoop环境
    4. 数据量不足时，实际分片数可能小于配置值
"""

# Python standard libraries
import base64
import os
import sys
from typing import Dict
import random
import json
import uuid
import traceback
import subprocess  # 确保导入 subprocess 模块
import gc
import time
import glob

# Third-party libraries
import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from pyspark.sql import SparkSession
from concurrent.futures import ThreadPoolExecutor
import math

# 调整图片尺寸到合适大小
def resize(image, max_height=640, max_width=640):
    """
    调整图片大小以适应指定的尺寸。
    参数:
        image (numpy.ndarray): 输入的图片，格式为numpy数组
        max_height (int): 最大高度
        max_width (int): 最大宽度
    返回:
        numpy.ndarray: 调整大小后的图片
    """
    # 获取图片的原始高度和宽度
    height, width = image.shape[:2]
    
    # 如果图片尺寸已经小于或等于目标尺寸，则直接返回原图片
    if height <= max_height and width <= max_width:
        return image
        
    # 计算缩放比例
    scale_h = max_height / height
    scale_w = max_width / width
    scale = min(scale_h, scale_w)
    
    # 计算新的尺寸
    new_height = int(height * scale)
    new_width = int(width * scale)
    
    # 调整图片大小
    return cv2.resize(image, (new_width, new_height))

# 定义方法将指定路径图片转为Base64编码
def encode_image(image_path):
  """
  将指定路径的图片进行编码
  参数:
      image_path (str): 图片文件的路径
  返回:
      str: 编码后的图片字符串
  """
  # 读取图片
  image = cv2.imread(image_path)
  # 调整图片大小
  image_resized = resize(image)
  # 将图片编码为JPEG格式
  _, encoded_image = cv2.imencode(".jpg", image_resized)
  # 将编码后的图片转换为Base64字符串
  return base64.b64encode(encoded_image).decode("utf-8")

class DataParser:
    def __init__(self, config):
        """初始化数据解析器"""
        self._init_config(config)
        self._init_data_sources()
        
    def _init_config(self, config):
        """初始化配置参数"""
        # 基础配置
        self.config = config
        self.image_folders = [folder.strip() for folder in config['image_folder'].split(';')]
        self.output_path = config['parquet_path']
        self.txt_file_path = config['txt_file_path']
        
        # 新增：支持读取txt目录
        self.txt_dir_path = config.get('txt_dir_path', '')
        
        # 新增：数据集分割结果保存路径
        self.split_output_dir = config.get('split_output_dir', '')
        
        # 数据源配置
        self.excel_file_path = config.get('excel_file_path', '')
        self.hdfs_path = config.get('hdfs_path', '')
        self.data_source = config.get('data_source', 'excel')
        
        # 输出配置
        self.num_shards = config.get('num_shards', 1)
        self.hdfs_output_path = config.get('hdfs_output_path', '')
        self.index_file = config.get('index_file', 'dataset_index.json')
        
        # 处理配置
        self.num_workers = config.get('num_workers', 1)
        self.source_name = config.get('source_name', 'wenjuan_photo_0207_1k')
        self.batch_size = config.get('batch_size', 1000)
        
        # 图片处理配置
        self.resize_image = config.get('resize_image', True)
        resize_config = config.get('resize_config', {})
        self.max_height = resize_config.get('max_height', 640)
        self.max_width = resize_config.get('max_width', 640)
        
        # 确保输出目录存在
        self._ensure_output_directory()
        
        # 用于收集生成的文件路径
        self.generated_files = []

    def _init_data_sources(self):
        """初始化数据源"""
        self.photo_id_to_images = self._get_photo_id_to_images()
        self.photo_texts = self._get_photo_texts()
        self.photo_cot_results = self._get_photo_cot_results()

    def _ensure_output_directory(self):
        """确保输出目录存在"""
        output_dir = os.path.dirname(self.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    def _process_single_image(self, image_path, photo_id, idx):
        """处理单张图片"""
        try:
            with open(image_path, 'rb') as img_file:
                image_bytes = img_file.read()
                image_key = f"{photo_id}_{idx}"
                return image_key, base64.b64encode(image_bytes).decode('utf-8')
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return None, None

    def _process_single_item(self, photo_id):
        """处理单个数据项，独立函数便于内存管理"""
        try:
            # 检查图片是否存在
            if photo_id not in self.photo_id_to_images:
                return None
                
            # 处理图片
            images = {}
            image_paths = self.photo_id_to_images[photo_id]
            for img_idx, image_path in enumerate(image_paths):
                image_key, image_data = self._process_single_image(image_path, photo_id, img_idx)
                if image_key and image_data:
                    images[image_key] = image_data
            
            if not images:
                return None
                
            # 构建消息数据
            messages = self._build_messages(photo_id, images)
            if not messages:
                return None
                
            # 返回处理后的数据项
            return {
                'images': json.dumps(images),
                'videos': json.dumps([]),
                'messages': json.dumps(messages),
                'segments': json.dumps(None),
                'source': self.source_name,
                'metadata': None,
                'uuid': str(uuid.uuid1())
            }
            
        except Exception as e:
            print(f"Error processing photo_id {photo_id}: {str(e)}")
            return None

    def _build_messages(self, photo_id, images):
        """构建消息数据"""
        if photo_id not in self.photo_texts:
            return None
            
        description = self.photo_texts[photo_id]
        user_text = self._build_user_text(description)
        satisfaction = "满意" if description['wenjuan_type'] == '问卷优质' else "不满意"
        
        # 检查是否有CoT结果
        assistant_content = self.photo_cot_results.get(photo_id, f"用户对视频的满意度结果是：{satisfaction}")
        
        return [
            {
                "role": "user", 
                "content": [
                    {
                        "type": "video",
                        "video": [
                            {"type": "image", "image": f"{photo_id}_{i}"} 
                            for i in range(len(images))
                        ]
                    },
                    {"type": "text", "text": user_text}
                ]
            },
            {
                "role": "assistant",
                "content": assistant_content
            }
        ]

    def _build_user_text(self, description):
        """构建用户文本"""
        return (
            f"你是一个短视频平台内容理解专家，为了将该视频推荐给潜在感兴趣的用户，"
            f"请结合短视频的视频抽帧结果、ocr 结果、asr 结果以及平台内所有用户对这个视频评论信息判断用户是否会对这个视频满意。"
            f"请充分考虑评论信息，结合这些信息提炼出该视频所有的看点。\n"
            f"对应的视频标题是：{description['caption']}\n"
            f"对应的视频 ocr 内容是：{description['ocr']}\n"
            f"对应的视频 asr 识别结果是：{description['asr']}\n"
            f"对应的短视频平台内用户评论内容是：{description['user_comment']}\n"
            f"请深呼吸并一步步仔细思考，请输出思考过程和结果。"
        )

    def _process_batch(self, photo_ids, batch_num, total_batches):
        """
        处理一批数据
        
        Args:
            photo_ids: List[str], 要处理的photo_id列表
            batch_num: int, 当前是第几个batch
            total_batches: int, 总batch数
        """
        print(f"\nProcessing batch {batch_num}/{total_batches} ({len(photo_ids)} photos)...")
        batch_data = []
        
        # 设置日志采样率：每处理20个项目打印一次日志
        log_interval = max(1, len(photo_ids) // 20)
        
        for idx, photo_id in enumerate(photo_ids, 1):
            try:
                # 检查图片是否存在
                if photo_id not in self.photo_id_to_images:
                    continue
                
                # 只在特定间隔打印日志
                should_log = idx % log_interval == 0
                if should_log:
                    print(f"  Batch {batch_num}/{total_batches} - Processing {idx}/{len(photo_ids)}: {photo_id}")
                
                # 处理单个图片并立即释放内存
                data_item = self._process_single_item(photo_id)
                if data_item:
                    batch_data.append(data_item)
                
                # 主动进行内存清理
                if idx % 100 == 0:
                    gc.collect()
                
            except Exception as e:
                if should_log:
                    print(f"    Error processing {photo_id}: {str(e)}")
                continue
        
        print(f"Batch {batch_num}/{total_batches} completed: {len(batch_data)}/{len(photo_ids)} items processed")
        return batch_data

    def _save_batch_to_temp(self, batch_data):
        """
        将批次数据保存到临时文件
        
        Args:
            batch_data: List[Dict], 批次数据
        Returns:
            str: 临时文件路径，失败则返回None
        """
        if not batch_data:
            return None
        
        # 使用时间戳和uuid确保文件名唯一
        timestamp = int(time.time() * 1000)
        unique_id = str(uuid.uuid4())
        temp_file = os.path.join(
            os.path.dirname(self.output_path),
            f"temp_{timestamp}_{unique_id}.parquet"
        )
        
        try:
            # 检查文件是否已存在
            if os.path.exists(temp_file):
                print(f"Warning: Temp file already exists: {temp_file}")
                return None
            
            # 构建表并立即写入文件
            schema = self._get_schema()
            arrays = [pa.array([item[field] for item in batch_data]) for field in schema.names]
            table = pa.Table.from_arrays(arrays, schema=schema)
            
            # 使用临时文件名先写入，成功后再重命名
            temp_file_writing = f"{temp_file}.writing"
            pq.write_table(table, temp_file_writing)
            
            # 原子性地重命名文件
            os.rename(temp_file_writing, temp_file)
            
            # 清理内存
            arrays = None
            table = None
            gc.collect()
            
            return temp_file
            
        except Exception as e:
            print(f"Error saving batch to temp file: {str(e)}")
            # 清理可能存在的临时文件
            for f in [temp_file, f"{temp_file}.writing"]:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except Exception:
                        pass
            return None

    def _get_schema(self):
        """获取数据schema"""
        return pa.schema([
            ('images', pa.string()),
            ('videos', pa.string()),
            ('messages', pa.string()),
            ('segments', pa.string()),
            ('source', pa.string()),
            ('metadata', pa.string()),
            ('uuid', pa.string())
        ])

    def _merge_parquet_files(self, temp_files, output_path, num_shards=None):
        """
        合并临时文件
        
        Args:
            temp_files: List[str], 临时文件路径列表
            output_path: str, 输出文件路径
            num_shards: Optional[int], 分片数量
        """
        if not temp_files:
            return
        
        try:
            tables = []
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    tables.append(pq.read_table(temp_file))
            
            if not tables:
                return
            
            merged_table = pa.concat_tables(tables)
            
            # 使用临时目录保存最终文件
            output_dir = os.path.dirname(output_path)
            temp_dir = os.path.join(output_dir, f"temp_merge_{int(time.time())}")
            os.makedirs(temp_dir, exist_ok=True)
            
            try:
                if num_shards and num_shards > 1:
                    self._save_sharded_output(merged_table, os.path.join(temp_dir, "temp"), num_shards)
                else:
                    temp_output = os.path.join(temp_dir, "temp.parquet")
                    pq.write_table(merged_table, temp_output)
                    
                # 移动文件到最终位置
                final_path = f"{output_path}.parquet"
                os.rename(temp_output, final_path)
                
            finally:
                # 清理临时文件
                for temp_file in temp_files:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                    
        except Exception as e:
            print(f"Error merging parquet files: {str(e)}")
            print(traceback.format_exc())

    def _save_sharded_output(self, table, output_path, num_shards):
        """
        保存分片输出
        
        Args:
            table: pyarrow.Table, 要保存的数据表
            output_path: str, 输出文件路径
            num_shards: int, 分片数量
        """
        total_rows = len(table)
        if total_rows == 0:
            print("Warning: No data to save")
            return
        
        # 计算每个分片的行数
        rows_per_shard = math.ceil(total_rows / num_shards)
        actual_num_shards = min(num_shards, math.ceil(total_rows / rows_per_shard))
        
        if actual_num_shards < num_shards:
            print(f"\nWarning: Not enough data to create {num_shards} shards.")
            print(f"Total rows: {total_rows}")
            print(f"Will create {actual_num_shards} shards instead.")
        
        # 使用临时文件进行分片写入
        temp_files = []
        try:
            for i in range(actual_num_shards):
                start_idx = i * rows_per_shard
                end_idx = min((i + 1) * rows_per_shard, total_rows)
                
                shard_table = table.slice(start_idx, end_idx - start_idx)
                temp_shard_path = f"{output_path}-{i:05d}-of-{actual_num_shards:05d}.parquet.tmp"
                final_shard_path = f"{output_path}-{i:05d}-of-{actual_num_shards:05d}.parquet"
                
                print(f"\nSaving shard {i+1}/{actual_num_shards}")
                print(f"  Rows: {len(shard_table)}")
                
                # 先写入临时文件
                pq.write_table(shard_table, temp_shard_path)
                temp_files.append((temp_shard_path, final_shard_path))
                
            # 所有分片都写入成功后，进行原子性重命名
            for temp_path, final_path in temp_files:
                os.rename(temp_path, final_path)
                if self.hdfs_output_path:
                    self._save_and_upload_file(final_path)
                
        except Exception as e:
            print(f"Error in sharded output: {str(e)}")
            # 清理临时文件
            for temp_path, _ in temp_files:
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass
            raise

    def _save_and_upload_file(self, file_path):
        """
        保存并上传文件
        
        Args:
            file_path: str, 文件保存路径
        """
        print(f"  Saving file locally: {file_path}")
        pq.write_table(file_path)
        
        if self.hdfs_output_path:
            hdfs_path = os.path.join(
                self.hdfs_output_path,
                os.path.basename(file_path)
            )
            print(f"  Uploading to HDFS: {hdfs_path}")
            if self._upload_to_hdfs(file_path, hdfs_path):
                self.generated_files.append(hdfs_path)
                print("  Upload successful")
                print(f"  Removing local file: {file_path}")
                os.remove(file_path)
            else:
                print("  Upload failed")

    def process_dataset(self):
        """处理整个数据集"""
        # 分割数据集
        train_ids, test_ids = self._split_dataset()
        
        # 处理训练集
        print("\nProcessing training set...")
        self._process_train_data(train_ids)
        
        # 处理测试集
        print("\nProcessing test set...")
        self._process_test_data(test_ids)

    def _process_train_data(self, train_ids):
        """处理训练集数据，直接分成num_shards份并行处理"""
        if not train_ids:
            print("No training data to process")
            return
        
        # 将数据均匀分成num_shards份
        shard_size = math.ceil(len(train_ids) / self.num_shards)
        shards = []
        for i in range(self.num_shards):
            start_idx = i * shard_size
            end_idx = min((i + 1) * shard_size, len(train_ids))
            shards.append(train_ids[start_idx:end_idx])
        
        print(f"Split training data into {self.num_shards} shards")
        print(f"Average shard size: {shard_size} items")
        
        # 创建工作线程池
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # 提交任务，每个任务处理一个分片
            futures = []
            for shard_idx, shard_data in enumerate(shards):
                future = executor.submit(
                    self._process_and_save_shard,
                    shard_data,
                    shard_idx,
                    self.num_shards,
                    "train"
                )
                futures.append(future)
            
            # 等待所有任务完成
            for future in futures:
                future.result()

    def _process_and_save_shard(self, photo_ids, shard_idx, total_shards, split_name):
        """处理并保存单个分片的数据"""
        print(f"\nProcessing {split_name} shard {shard_idx + 1}/{total_shards}")
        print(f"Items in shard: {len(photo_ids)}")
        
        # 处理该分片的所有数据
        processed_data = []
        log_interval = max(1, len(photo_ids) // 20)  # 5%的数据打一次日志
        
        for idx, photo_id in enumerate(photo_ids, 1):
            try:
                if idx % log_interval == 0:
                    print(f"  Shard {shard_idx + 1}/{total_shards} - "
                          f"Processing {idx}/{len(photo_ids)}")
                
                item = self._process_single_item(photo_id)
                if item:
                    processed_data.append(item)
                
                # 定期进行内存清理
                if idx % 100 == 0:
                    gc.collect()
                
            except Exception as e:
                print(f"Error processing {photo_id}: {str(e)}")
                continue
        
        # 直接保存为对应的分片文件
        if processed_data:
            output_name = os.path.splitext(self.output_path)[0]
            if split_name == "train":
                output_file = f"{output_name}-{split_name}-{shard_idx:05d}-of-{total_shards:05d}.parquet"
            else:
                output_file = f"{output_name}-{split_name}.parquet"
            
            print(f"\nSaving {split_name} shard {shard_idx + 1}/{total_shards}")
            print(f"Processed items: {len(processed_data)}")
            
            # 保存文件
            table = pa.Table.from_pylist(processed_data)
            pq.write_table(table, output_file)
            
            # 如果配置了HDFS路径，上传到HDFS
            if self.hdfs_output_path:
                hdfs_path = os.path.join(
                    self.hdfs_output_path,
                    os.path.basename(output_file)
                )
                if self._upload_to_hdfs(output_file, hdfs_path):
                    self.generated_files.append(hdfs_path)
                    os.remove(output_file)  # 上传成功后删除本地文件

    def _process_test_data(self, test_ids):
        """处理测试集数据，保存为单个文件"""
        if not test_ids:
            print("No test data to process")
            return
        
        # 测试集直接用一个线程处理
        self._process_and_save_shard(test_ids, 0, 1, "test")

    def _split_dataset(self):
        """分割数据集"""
        all_photo_ids = list(self.photo_id_to_images.keys())
        random.shuffle(all_photo_ids)
        
        split_idx = int(len(all_photo_ids) * 0.9)
        train_ids = all_photo_ids[:split_idx]
        test_ids = all_photo_ids[split_idx:]
        
        print(f"Total photos: {len(all_photo_ids)}")
        print(f"Train set size: {len(train_ids)}")
        print(f"Test set size: {len(test_ids)}")
        
        # 如果配置了分割结果输出目录，保存分割结果
        if self.split_output_dir:
            self._save_split_results(train_ids, test_ids)
        
        return train_ids, test_ids

    def _save_split_results(self, train_ids, test_ids):
        """
        保存数据集分割结果
        
        Args:
            train_ids: 训练集ID列表
            test_ids: 测试集ID列表
        """
        try:
            # 确保输出目录存在
            os.makedirs(self.split_output_dir, exist_ok=True)
            
            # 构建输出文件路径
            train_file = os.path.join(self.split_output_dir, "train_ids.txt")
            test_file = os.path.join(self.split_output_dir, "test_ids.txt")
            
            # 保存训练集ID
            with open(train_file, 'w', encoding='utf-8') as f:
                f.write("photo_id\n")  # 写入表头
                for photo_id in sorted(train_ids):  # 排序以保持稳定性
                    f.write(f"{photo_id}\n")
            
            # 保存测试集ID
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write("photo_id\n")  # 写入表头
                for photo_id in sorted(test_ids):  # 排序以保持稳定性
                    f.write(f"{photo_id}\n")
            
            print(f"\nSplit results saved:")
            print(f"Train IDs: {train_file}")
            print(f"Test IDs: {test_file}")
            
        except Exception as e:
            print(f"Error saving split results: {str(e)}")
            print(traceback.format_exc())

    def _get_photo_id_to_images(self):
        """获取图片ID到图片文件的映射，递归搜索所有子文件夹"""
        photo_id_to_images = {}
        
        for folder in self.image_folders:
            if not os.path.exists(folder):
                print(f"Warning: Image folder {folder} does not exist")
                continue
                
            # 递归遍历文件夹
            for root, _, files in os.walk(folder):
                # 对每个photo_id的文件进行排序，确保顺序一致
                image_files = [f for f in files if f.endswith('.jpg')]
                for file in sorted(image_files):  # 添加排序
                    photo_id = file.split('_')[0]
                    full_path = os.path.join(root, file)
                    if photo_id in photo_id_to_images:
                        photo_id_to_images[photo_id].append(full_path)
                    else:
                        photo_id_to_images[photo_id] = [full_path]
        
        # 确保每个photo_id下的图片列表都是有序的
        for photo_id in photo_id_to_images:
            photo_id_to_images[photo_id].sort()
        
        print(f"Found {len(photo_id_to_images)} unique photo IDs across all subfolders")
        return photo_id_to_images

    def _get_photo_texts(self):
        """根据配置的数据源获取文本数据"""
        if self.data_source == 'hdfs':
            if not self.hdfs_path:
                print("Error: HDFS path not configured")
                return {}
            return self._read_from_hdfs()
        else:  # default to excel
            if not self.excel_file_path:
                print("Error: Excel file path not configured")
                return {}
            return self._read_from_excel()

    def _read_from_excel(self):
        """从Excel文件获取文本数据"""
        try:
            print(f"\nTrying to read Excel file: {self.excel_file_path}")
            if not os.path.exists(self.excel_file_path):
                print("File does not exist!")
                return {}
            
            print(f"File size: {os.path.getsize(self.excel_file_path)} bytes")
            
            # 尝试使用 openpyxl 读取
            try:
                print("Trying openpyxl engine...")
                df = pd.read_excel(self.excel_file_path, engine='openpyxl')
            except Exception as e1:
                print(f"openpyxl failed: {str(e1)}")
                try:
                    # 如果是 .xls 文件，尝试转换为 xlsx
                    print("Converting xls to xlsx...")
                    import subprocess
                    xlsx_path = self.excel_file_path.replace('.xls', '.xlsx')
                    
                    # 使用 libreoffice 转换文件格式
                    try:
                        subprocess.run(['libreoffice', '--headless', '--convert-to', 'xlsx', 
                                     self.excel_file_path, '--outdir', 
                                     os.path.dirname(self.excel_file_path)], 
                                    check=True)
                        print(f"Converted to: {xlsx_path}")
                        df = pd.read_excel(xlsx_path, engine='openpyxl')
                    except Exception as e2:
                        print(f"Conversion failed: {str(e2)}")
                        # 如果转换失败，尝试使用其他方法
                        try:
                            print("Trying default engine...")
                            df = pd.read_excel(self.excel_file_path)
                        except Exception as e3:
                            print(f"All methods failed: {str(e3)}")
                            return {}

                except Exception as e4:
                    print(f"Failed to process file: {str(e4)}")
                    return {}

            print(f"\nSuccessfully read Excel file with {len(df)} rows")
            print(f"Columns: {df.columns.tolist()}")
            print("\nFirst few rows:")
            print(df.head())
            
            # 确保必要的列存在
            required_columns = ['photo_id', 'wenjuan_type', 'caption', 'user_comment', 'ocr', 'asr']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"\nWarning: Missing required columns: {missing_columns}")
                print("Available columns:", df.columns.tolist())
                return {}

            # 处理数据
            photo_texts = {}
            for index, row in df.iterrows():
                try:
                    photo_id = str(row['photo_id']).strip()
                    if not photo_id or pd.isna(photo_id):
                        continue
                        
                    photo_texts[photo_id] = {
                        "wenjuan_type": str(row['wenjuan_type']) if not pd.isna(row['wenjuan_type']) else "",
                        "caption": str(row['caption']) if not pd.isna(row['caption']) else "",
                        "user_comment": str(row['user_comment']) if not pd.isna(row['user_comment']) else "",
                        "ocr": str(row['ocr']) if not pd.isna(row['ocr']) else "",
                        "asr": str(row['asr']) if not pd.isna(row['asr']) else ""
                    }
                except Exception as e:
                    print(f"Error processing row {index}: {str(e)}")
                    continue

            print(f"\nSuccessfully processed {len(photo_texts)} photo texts")
            if photo_texts:
                sample_ids = list(photo_texts.keys())[:3]
                print(f"Sample photo_ids: {sample_ids}")
                print("Sample data for first photo_id:")
                print(json.dumps(photo_texts[sample_ids[0]], indent=2, ensure_ascii=False))
            
            return photo_texts
            
        except Exception as e:
            print(f"Error reading Excel file: {str(e)}")
            return {}

    def _get_photo_cot_results(self):
        """获取CoT结果"""
        responses = {}
        
        # 如果配置了txt目录，优先从目录读取
        if self.txt_dir_path and os.path.isdir(self.txt_dir_path):
            print(f"\nReading CoT results from directory: {self.txt_dir_path}")
            import re
            content_pattern = re.compile(r"'content': '(.*?)'(?=, 'tool_calls')")
            processed_file_num = 0
            
            for txt_file in glob.glob(os.path.join(self.txt_dir_path, "*.txt")):
                try:
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if "photo_id is" in line and "response is" in line:
                                try:
                                    photo_id = line.split("photo_id is")[1].split(",")[0].strip()
                                    # 使用正则表达式直接提取content
                                    match = content_pattern.search(line)
                                    if match:
                                        content = match.group(1)
                                        responses[photo_id] = content
                                except Exception as e:
                                    print(f"处理行时出错: {e}")
                                    continue
                except Exception as e:
                    print(f"处理文件 {txt_file} 时出错: {e}")
                    continue
                processed_file_num += 1
                if processed_file_num % 1000 == 0:
                    print(f"已处理 {processed_file_num} 个 cot 文件")
            
            print(f"Loaded {len(responses)} CoT results from directory")
            
        # 如果还配置了单个txt文件，也读取它
        elif os.path.exists(self.txt_file_path):
            try:
                with open(self.txt_file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if "photo_id is" in line and "response is" in line:
                            try:
                                photo_id = line.split("photo_id is")[1].split(",")[0].strip()
                                match = content_pattern.search(line)
                                if match:
                                    content = match.group(1)
                                    responses[photo_id] = content
                            except Exception as e:
                                print(f"处理行时出错: {e}")
                                continue
            except Exception as e:
                print(f"处理文件 {self.txt_file_path} 时出错: {e}")
        
        return responses

    def _read_from_hdfs(self):
        """从HDFS读取数据"""
        try:
            import subprocess
            import tempfile
            
            # 创建临时目录来存储从HDFS下载的文件
            with tempfile.TemporaryDirectory() as temp_dir:
                print(f"\nCreated temporary directory: {temp_dir}")
                print(f"Downloading data from HDFS: {self.hdfs_path}")
                
                # 使用hadoop命令下载文件
                cmd = f"hadoop fs -get {self.hdfs_path}/* {temp_dir}"
                process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                
                if process.returncode != 0:
                    print(f"Error downloading files from HDFS: {process.stderr}")
                    return {}
                    
                print("Successfully downloaded files from HDFS")
                
                # 读取所有下载的文件
                photo_texts = {}
                
                # 跳过_SUCCESS文件
                for filename in os.listdir(temp_dir):
                    if filename == '_SUCCESS':
                        continue
                        
                    file_path = os.path.join(temp_dir, filename)
                    print(f"Processing file: {filename}")
                    
                    try:
                        # 尝试使用pandas读取parquet文件
                        df = pd.read_parquet(file_path)
                        
                        if df is None or len(df) == 0:
                            print(f"Empty dataframe for file: {filename}")
                            continue
                            
                        print(f"Columns in file: {df.columns.tolist()}")
                        
                        # 确保必要的列存在
                        required_columns = ['photo_id', 'wenjuan_type', 'caption', 'user_comment', 'ocr', 'asr']
                        missing_columns = [col for col in required_columns if col not in df.columns]
                        if missing_columns:
                            print(f"Warning: Missing required columns in {filename}: {missing_columns}")
                            print(f"Available columns: {df.columns.tolist()}")
                            continue
                        
                        # 安全地处理每一行数据
                        def safe_get_str(value):
                            """安全地获取字符串值，处理所有可能的空值情况"""
                            if isinstance(value, (list, np.ndarray)):
                                # 如果是数组，取第一个非空值
                                for v in value:
                                    if v and not pd.isna(v):
                                        return str(v)
                                return ""
                            elif pd.isna(value) or value is None:
                                return ""
                            else:
                                return str(value)
                        
                        # 处理每一行数据
                        for idx, row in df.iterrows():
                            photo_id = safe_get_str(row['photo_id'])
                            if not photo_id:  # 跳过空的photo_id
                                continue
                                
                            photo_texts[photo_id] = {
                                "wenjuan_type": safe_get_str(row['wenjuan_type']),
                                "caption": safe_get_str(row['caption']),
                                "user_comment": safe_get_str(row['user_comment']),
                                "ocr": safe_get_str(row['ocr']),
                                "asr": safe_get_str(row['asr'])
                            }
                            
                            # 每处理1000条数据打印一次进度
                            if len(photo_texts) % 1000 == 0:
                                print(f"Processed {len(photo_texts)} records...")
                                
                    except Exception as e:
                        print(f"Error processing file {filename}: {str(e)}")
                        print("Stack trace:", traceback.format_exc())
                        continue
                
                print(f"\nSuccessfully processed {len(photo_texts)} photo texts from HDFS")
                if photo_texts:
                    # 打印一个样本数据
                    sample_id = next(iter(photo_texts))
                    print("\nSample data:")
                    print(f"Photo ID: {sample_id}")
                    print(json.dumps(photo_texts[sample_id], indent=2, ensure_ascii=False))
                
                return photo_texts
                
        except Exception as e:
            print(f"Error reading from HDFS: {str(e)}")
            print("Stack trace:", traceback.format_exc())
            return {}

    def _upload_to_hdfs(self, local_path, hdfs_path):
        """
        将本地文件上传到HDFS
        Args:
            local_path: 本地文件路径
            hdfs_path: HDFS目标路径
        Returns:
            bool: 上传是否成功
        """
        try:
            if not os.path.exists(local_path):
                print(f"Error: Local file does not exist: {local_path}")
                return False
            
            # 确保HDFS目标目录存在
            hdfs_dir = os.path.dirname(hdfs_path)
            mkdir_cmd = f"hadoop fs -mkdir -p {hdfs_dir}"
            process = subprocess.run(mkdir_cmd, shell=True, capture_output=True, text=True)
            if process.returncode != 0:
                print(f"Error creating HDFS directory: {process.stderr}")
                return False

            # 上传文件到HDFS
            cmd = f"hadoop fs -put -f {local_path} {hdfs_path}"
            process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if process.returncode != 0:
                print(f"Error uploading file to HDFS: {process.stderr}")
                return False
            
            # 验证文件是否成功上传
            check_cmd = f"hadoop fs -test -e {hdfs_path}"
            process = subprocess.run(check_cmd, shell=True)
            if process.returncode != 0:
                print(f"Error: File not found in HDFS after upload: {hdfs_path}")
                return False
            
            print(f"Successfully uploaded file to HDFS: {hdfs_path}")
            return True
            
        except Exception as e:
            print(f"Error during HDFS upload: {str(e)}")
            print("Stack trace:", traceback.format_exc())
            return False

    def _write_index_file(self):
        """将生成的文件路径写入索引文件"""
        try:
            if not self.generated_files:
                print("No files to index")
                return
            
            # 确保文件按照固定顺序排序
            self.generated_files.sort()
            
            # 过滤掉测试集的文件
            filtered_files = [file for file in self.generated_files if "test" not in file]

            # 写入JSON文件（单行格式）
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(filtered_files, f, ensure_ascii=False, separators=(',', ':'))
            
            print(f"Successfully wrote {len(filtered_files)} file paths to {self.index_file}")
            
            # 如果配置了HDFS输出路径，也将索引文件上传到HDFS
            if self.hdfs_output_path:
                hdfs_index_path = os.path.join(
                    self.hdfs_output_path,
                    os.path.basename(self.index_file)
                )
                if self._upload_to_hdfs(self.index_file, hdfs_index_path):
                    print(f"Uploaded index file to HDFS: {hdfs_index_path}")
                else:
                    print("Failed to upload index file to HDFS")
                    
        except Exception as e:
            print(f"Error writing index file: {str(e)}")
            print("Stack trace:", traceback.format_exc())

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python parse_data_to_parquet.py <config.yaml>")
        sys.exit(1)

    config_path = sys.argv[1]
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    parser = DataParser(config)
    parser.process_dataset()