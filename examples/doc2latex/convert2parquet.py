import os
import base64
import json
import uuid
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from tqdm import tqdm

def encode_image_to_base64(image_path):
    """将图片转换为base64编码"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def process_image_dataset(folder_path, output_dir, batch_size=1000):
    """处理图片数据集并保存为parquet文件"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有png文件
    folder = Path(folder_path)
    png_files = list(folder.rglob("*.[pP][nN][gG]"))
    
    # 添加其他常见图片格式
    jpg_files = list(folder.rglob("*.[jJ][pP][gG]")) + list(folder.rglob("*.[jJ][pP][eE][gG]"))
    all_image_files = png_files + jpg_files
    
    records = []
    batch_counter = 0
    file_counter = 0
    
    for img_path in tqdm(all_image_files):
        # 为每个图片生成一个uuid
        sample_uuid = str(uuid.uuid4())
        
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
        
        # 每当记录数达到batch_size，保存为一个parquet文件
        if batch_counter >= batch_size:
            df = pd.DataFrame(records)
            table = pa.Table.from_pandas(df)
            output_file = os.path.join(output_dir, f"{file_counter}.parquet")
            pq.write_table(table, output_file)
            
            print(f"已保存parquet文件: {output_file}，包含 {batch_counter} 条记录")
            
            # 重置记录和计数器
            records = []
            batch_counter = 0
            file_counter += 1
    
    # 保存剩余的记录
    if records:
        df = pd.DataFrame(records)
        table = pa.Table.from_pandas(df)
        output_file = os.path.join(output_dir, f"{file_counter}.parquet")
        pq.write_table(table, output_file)
        print(f"已保存parquet文件: {output_file}，包含 {batch_counter} 条记录")
    
    print(f"处理完成，共处理 {len(all_image_files)} 个图片，生成 {file_counter + 1} 个parquet文件")

if __name__ == "__main__":
    # 设置输入图片文件夹和输出parquet文件夹
    input_folder = "/mmu_nlp_hdd/zhouguorui/pdf2png_process/output"  # 替换为您的图片文件夹路径
    output_folder = "/mmu_nlp_hdd/zhouyang12/data/pdf2png_process/parquet"  # 替换为输出parquet文件的文件夹路径
    
    # 处理图片数据集
    process_image_dataset(input_folder, output_folder, batch_size=256)