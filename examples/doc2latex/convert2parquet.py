import os
import base64
import json
import uuid
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import multiprocessing
from functools import partial
from tqdm import tqdm
import time

def encode_image_to_base64(image_path):
    """将图片转换为base64编码"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def process_image_chunk(chunk, output_dir, process_id, batch_size=1000):
    """处理一个图片块并保存为parquet文件"""
    records = []
    batch_counter = 0
    file_counter = 0
    files_written = []
    
    # 进度条
    pbar = tqdm(total=len(chunk), desc=f"进程 {process_id}", position=process_id)
    
    for img_path in chunk:
        # 为每个图片生成一个uuid
        sample_uuid = str(uuid.uuid4())
        
        try:
            # 将图片编码为base64
            img_base64 = encode_image_to_base64(img_path)
            
            # 构建image key，使用简单的数字命名
            image_key = "0.jpg"
            
            # 构建images字段（map<string, string>）
            images_map = {image_key: img_base64}
            images_json = json.dumps(images_map)
            
            # 构建messages字段
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_key},
                        {"type": "text", "text": "识别图片中的内容，输出为LaTeX格式"},
                    ],
                }
            ]
            messages_json = json.dumps(messages)
            
            # 构建其他字段
            record = {
                "images": images_json,
                "videos": json.dumps([]),  # 空列表，因为没有视频
                "source": str(img_path),   # 图片的原始路径
                "messages": messages_json,
                "segments": json.dumps([]),  # 空的pretrain数据
                "metadata": json.dumps({}),  # 空的metadata
                "uuid": sample_uuid
            }
            
            records.append(record)
            batch_counter += 1
        except Exception as e:
            print(f"处理图片 {img_path} 时出错: {e}")
        
        # 更新进度条
        pbar.update(1)
        
        # 每当记录数达到batch_size，保存为一个parquet文件
        if batch_counter >= batch_size:
            df = pd.DataFrame(records)
            table = pa.Table.from_pandas(df)
            output_file = os.path.join(output_dir, f"{process_id}_{file_counter}.parquet")
            pq.write_table(table, output_file)
            
            files_written.append((output_file, batch_counter))
            
            # 重置记录和计数器
            records = []
            batch_counter = 0
            file_counter += 1
    
    # 保存剩余的记录
    if records:
        df = pd.DataFrame(records)
        table = pa.Table.from_pandas(df)
        output_file = os.path.join(output_dir, f"{process_id}_{file_counter}.parquet")
        pq.write_table(table, output_file)
        files_written.append((output_file, batch_counter))
    
    pbar.close()
    return files_written

def process_image_dataset_parallel(folder_path, output_dir, num_processes=None, batch_size=1000):
    """使用多进程并行处理图片数据集并保存为parquet文件"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    
    print(f"使用 {num_processes} 个进程并行处理图片...")
    
    # 获取所有图片文件
    start_time = time.time()
    folder = Path(folder_path)
    print("正在扫描所有图片文件...")
    
    png_files = list(folder.rglob("*.[pP][nN][gG]"))
    jpg_files = list(folder.rglob("*.[jJ][pP][gG]")) + list(folder.rglob("*.[jJ][pP][eE][gG]"))
    all_image_files = png_files + jpg_files
    
    print(f"找到 {len(all_image_files)} 个图片文件，用时 {time.time() - start_time:.2f} 秒")
    
    # 分割图片列表为几个大致相等的块
    chunk_size = len(all_image_files) // num_processes
    chunks = [all_image_files[i:i + chunk_size] for i in range(0, len(all_image_files), chunk_size)]
    
    # 创建进程池并处理图片块
    start_time = time.time()
    pool = multiprocessing.Pool(processes=num_processes)
    
    # 创建偏函数，固定output_dir和batch_size参数
    process_func = partial(process_image_chunk, output_dir=output_dir, batch_size=batch_size)
    
    # 为每个进程分配一个进程ID
    process_args = [(chunks[i], i) for i in range(len(chunks))]
    
    # 启动多进程处理
    results = pool.starmap(process_func, process_args)
    
    # 关闭进程池
    pool.close()
    pool.join()
    
    # 汇总结果
    total_files = 0
    total_records = 0
    for proc_results in results:
        for output_file, record_count in proc_results:
            total_files += 1
            total_records += record_count
    
    print(f"处理完成，共处理 {len(all_image_files)} 个图片，生成 {total_files} 个parquet文件，总计 {total_records} 条记录")
    print(f"总处理时间: {time.time() - start_time:.2f} 秒")

if __name__ == "__main__":
    # 设置输入图片文件夹和输出parquet文件夹
    input_folder = "/mmu_nlp_hdd/zhouguorui/pdf2png_process/output"  # 替换为您的图片文件夹路径
    output_folder = "/mmu_nlp_hdd/zhouyang12/data/pdf2png_process/parquet"  # 替换为输出parquet文件的文件夹路径
    
    # 设置进程数，默认是CPU核心数，可以根据需要调整
    num_processes = multiprocessing.cpu_count()
    
    # 处理图片数据集
    process_image_dataset_parallel(input_folder, output_folder, num_processes=num_processes, batch_size=512)