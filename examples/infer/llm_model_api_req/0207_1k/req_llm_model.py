import base64
import os
import cv2
import pandas as pd
import sys
import yaml
from typing import Optional
from openai import OpenAI
from volcenginesdkarkruntime import Ark
import warnings
import concurrent.futures

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

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

# 构建messages列表并发送请求
def process_photo(photo_id, image_files, photo_texts, client, image_folder, config):
    messages = []
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

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        base64_image = encode_image(image_path)
        context.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
                "detail": "high"
            },
        })

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
    print("prompt is ", config["prompt_pre"], config["prompt_end"])

    try:
        response = client.chat.completions.create(
            model=config['model_name'],
            messages=messages,
        )
        return photo_id, response
    except Exception as e:
        print(f"API request failed for photo_id {photo_id}: {e}")
        return photo_id, None

def get_client(client_type, api_key):
    if client_type in ("doubao", "ark"):
        client = Ark(
            api_key=api_key,
        )
    elif client_type in ('qwen'):
        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    return client

def main(config_path, api_key):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    client_type = config['client_type']
    folder_path = config['folder_path']
    client = get_client(client_type, api_key)

    image_folder = os.path.join(folder_path, config['image_folder'])
    excel_file_path = os.path.join(folder_path, config['excel_file_path'])
    is_debug = config['is_debug']

    photo_id_to_images = read_img_data(image_folder)
    photo_texts = read_csv_data(excel_file_path)

    response_list = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 如果是调试模式，只处理第一个 photo_id
        if is_debug:
            photo_id, image_files = next(iter(photo_id_to_images.items()))
            future = executor.submit(process_photo, photo_id, image_files, photo_texts, client, image_folder, config)
            photo_id, response = future.result()
            print("photo_id is ", photo_id, ", response is ", response)
        else:
            futures = {executor.submit(process_photo, photo_id, image_files, photo_texts, client, image_folder, config): photo_id for photo_id, image_files in photo_id_to_images.items()}
            for future in concurrent.futures.as_completed(futures):
                photo_id = futures[future]
                print("end process photo: ", photo_id)
                try:
                    photo_id, response = future.result()
                    if response:
                        response_list[photo_id] = response
                except Exception as e:
                    print(f"Photo ID {photo_id} generated an exception: {e}")
    if not is_debug:
        response_file_path = config['result_file']
        with open(response_file_path, 'w', encoding='utf-8') as f:
            for photo_id, response in response_list.items():
                f.write(f"photo_id is {photo_id}, response is {response}\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python req_llm_model.py config.yaml api_key")
        sys.exit(1)
    config_path = sys.argv[1]
    api_key = sys.argv[2]
    main(config_path, api_key)