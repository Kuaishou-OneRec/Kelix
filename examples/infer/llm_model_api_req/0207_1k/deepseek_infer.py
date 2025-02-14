import requests
import base64
import os
# 通过 pip install volcengine-python-sdk[ark] 安装方舟SDK
# from volcenginesdkarkruntime import Ark
import cv2
from typing import Optional
# 导入时间库，用于计时
import time
# 导入文件操作库，用于删除目录
import shutil
# 导入枚举库，用于定义抽帧策略
from enum import Enum
import pandas as pd
from openai import OpenAI
import concurrent.futures
import yaml
import sys

# 调整图片尺寸到合适大小
def resize(image):
    height, width = image.shape[:2]
    if height < width:
        target_height, target_width = 480, 640
    else:
        target_height, target_width = 640, 480
    if height <= target_height and width <= target_width:
        return image
    if height / target_height < width / target_width:
        new_width = target_width
        new_height = int(height * (new_width / width))
    else:
        new_height = target_height
        new_width = int(width * (new_height / height))
    return cv2.resize(image, (new_width, new_height))
  
# 定义方法将指定路径图片转为Base64编码
def encode_image(image_path):
  image = cv2.imread(image_path)
  image_resized = resize(image)
  _, encoded_image = cv2.imencode(".jpg", image_resized)
  return base64.b64encode(encoded_image).decode("utf-8")

def read_img_data(image_folder):
    # 获取文件夹中的所有jpg文件
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]

    # 按photo_id分组
    photo_id_to_images = {}
    for image_file in image_files:
        photo_id = image_file.split('_')[0]
        if photo_id not in photo_id_to_images:
            photo_id_to_images[photo_id] = []
        photo_id_to_images[photo_id].append(image_file)
    return photo_id_to_images

def read_csv_data(excel_file_path):
    # 读取Excel文件并解析
    df = pd.read_excel(excel_file_path)
    photo_texts = {}
    for index, row in df.iterrows():
        photo_id = str(row['photo_id'])
        photo_texts[photo_id] = {
            "wenjuan_type": str(row['wenjuan_type']) if not pd.isna(row['wenjuan_type']) else "",
            "caption": str(row['caption']) if not pd.isna(row['caption']) else "",
            "user_comment": str(row['user_comment']) if not pd.isna(row['user_comment']) else "",
            "ocr": str(row['ocr']) if not pd.isna(row['ocr']) else "",
            "asr": str(row['asr']) if not pd.isna(row['asr']) else ""
        }
    return photo_texts

def process_photo(photo_id, image_files, photo_texts, url, folder_path, config):
    messages = []
    print("start process photo ", photo_id)

    # 构建描述
    if photo_id in photo_texts:
        description = photo_texts[photo_id]
        context = [
            {"type": "text", "text": "对应的视频标题是：" + description["caption"]},
            {"type": "text", "text": "对应的视频 ocr 内容是：" + description["ocr"]},
            {"type": "text", "text": "对应的视频 asr 识别结果是：" + description["asr"]},
        ]
        if config["enable_cmt"]:
            context.append({"type": "text", "text": "对应的短视频平台内用户评论内容是：" + description["user_comment"]})
    else:
        context = []

    messages.append({
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": config["prompt_pre"],
            },
            *context,
            {
                "type": "text",
                "text": config["prompt_end"],
            }
        ],
    })
    json_data = {
        "model": "/llm_reco_ssd/zhouyang12/models/DeepSeek-R1/",
        "messages": messages
    }
    if config['is_debug']:
        print("prompt is ", messages)
    response = requests.post(url, json=json_data)
    if response.status_code == 200:
        print("end process ", photo_id, ", response is ", response.json())
        response_file_path = config['output_filename']
        with open(response_file_path, 'a', encoding='utf-8') as f:
            f.write(f"photo_id is {photo_id}, response is {response.json()}\n")
    else:
        print(f"请求失败，状态码: {response.status_code}")


def main(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    folder_path = config['folder_path']
    image_folder = os.path.join(folder_path, config['image_folder'])
    excel_file_path = os.path.join(folder_path, config['excel_file_path'])
    is_debug = config['is_debug']
    url = config['url']

    photo_id_to_images = read_img_data(image_folder)
    photo_texts = read_csv_data(excel_file_path)

    with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(process_photo, photo_id, image_files, photo_texts, url, folder_path, config)
                for photo_id, image_files in photo_id_to_images.items()
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python chatgpt_client.py config.yaml")
        sys.exit(1)
    config_filename = sys.argv[1]
    main(config_filename)