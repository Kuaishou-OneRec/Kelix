#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys,os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import logging
import time
import yaml
from google.protobuf import text_format

from kess.framework import (
    ClientOption,
    GrpcClient,
    KessOption,
)

from mmu.mmu_chat_gpt_pb2 import MmuChatGptRequest,MmuChatGptResponse
from mmu.mmu_chat_gpt_pb2_grpc import (
    MmuChatGptServiceStub,
)
from mmu.media_common_pb2 import ImgUnit
import cv2
from typing import Optional
# 导入文件操作库，用于删除目录
import shutil
# 导入枚举库，用于定义抽帧策略
from enum import Enum
from openai import OpenAI
import pandas as pd
import concurrent.futures

logger = logging.getLogger(__name__)
fmt_str = ('%(asctime)s.%(msecs)03d %(levelname)7s '
           '[%(thread)d][%(process)d] %(message)s')
fmt = logging.Formatter(fmt_str, datefmt='%H:%M:%S')
handler = logging.StreamHandler()
handler.setFormatter(fmt)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
folder_path = "/home/huqigen03"
image_folder = os.path.join(folder_path, "photo_0207_1k/4fd811bcd31248778b219127ce4639da/")
excel_file_path = os.path.join(folder_path, "0207_1k.xlsx")
print("excel_file_path is ", excel_file_path)

def chat(grpc_client: GrpcClient, timeout: float):

    try:
        #构造request, biz需根据实际申请的进行修改
        biz = 'test3.5'
        request = MmuChatGptRequest(biz=biz)
        request.session_id = 'test'
        request.req_id = '1000'
        request.query = 'chatGPT是什么'
        #发起请求
        resp = grpc_client.Chat(request, timeout=timeout)
        #打印结果
        logger.info(text_format.MessageToString(resp, as_utf8=True))

    except Exception as e:
        logger.error('发生异常, err: %s', e)


def emb(grpc_client: GrpcClient, timeout: float):

    try:
        #构造request, biz需根据实际申请的进行修改
        biz = 'test'
        request = MmuChatGptRequest(biz=biz)
        request.session_id = 'test'
        request.req_id = '1000'
        request.query = 'chatGPT是什么'
        #发起请求
        resp = grpc_client.Emb(request, timeout=timeout)
        #打印结果
        logger.info(text_format.MessageToString(resp, as_utf8=True))

    except Exception as e:
        logger.error('发生异常, err: %s', e)

def vision(grpc_client: GrpcClient, timeout: float, prompt, photo_list, photo_id, sid_prefix, output_filename):
    response = {}
    try:
        #构造request, biz需根据实际申请的进行修改
        biz = 'luoxinchen_b585bcc7_gpt-4o-2024-08-06'
        request = MmuChatGptRequest(biz=biz)
        request.session_id = sid_prefix + str(photo_id)
        request.req_id = sid_prefix+str(photo_id)
        request.query = prompt
        # request.config['ptu_only']='True'

        for photo in photo_list:
            with open(photo, 'rb') as f:
                image_data = f.read()
            img = ImgUnit(image = image_data)
            request.img.append(img)

        #发起请求
        resp = grpc_client.Chat(request, timeout=timeout)
        response[photo_id] = resp.answer
        with open(output_filename, 'a', encoding='utf-8') as f:
            f.write(f"photo_id is {photo_id}, response is {resp.answer}\n")
        print("end process photo : ", photo_id)

    except Exception as e:
        logger.error('发生异常, err: %s', e)
        
# 调整图片尺寸到合适大小
def resize(image):
    """
    调整图片大小以适应指定的尺寸。
    参数:
        image (numpy.ndarray): 输入的图片，格式为numpy数组。
    返回:
        numpy.ndarray: 调整大小后的图片。
    """
    # 获取图片的原始高度和宽度
    height, width = image.shape[:2]
    # 根据图片的宽高比确定目标尺寸
    if height < width:
        target_height, target_width = 480, 640
    else:
        target_height, target_width = 640, 480
    # 如果图片尺寸已经小于或等于目标尺寸，则直接返回原图片
    if height <= target_height and width <= target_width:
        return image
    # 计算新的高度和宽度，保持图片的宽高比
    if height / target_height < width / target_width:
        new_width = target_width
        new_height = int(height * (new_width / width))
    else:
        new_height = target_height
        new_width = int(width * (new_height / height))
    # 调整图片大小
    return cv2.resize(image, (new_width, new_height))

def process_photo(photo_id, image_files, photo_texts, client, config):
    # 获取对应的文字描述
    if photo_id in photo_texts:
        description = photo_texts[photo_id]
        prompt = config["prompt_pre"] + ",对应的视频标题是" + description["caption"] + ",对应的视频 ocr 内容是：" + description["ocr"] + ",对应的视频 asr 识别结果是：" + description["asr"]
        if config["enable_cmt"]:
         prompt += ",对应的短视频平台内用户评论内容是：" + description["user_comment"]
    else:
        prompt = config["prompt_pre"]
    prompt += config["prompt_end"]

    vision(client, 180, prompt, image_files, photo_id, config["sid_prefix"], config["output_filename"])
  
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

def rate_limited_submit(executor, func, *args, **kwargs):
    # 提交任务
    future = executor.submit(func, *args, **kwargs)
    # 添加延迟，限制速率
    time.sleep(30)  # 这里设置为每30秒提交一个任务，可以根据需要调整
    return future

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python chatgpt_client.py config.yaml")
        sys.exit(1)

    config_filename = sys.argv[1]

    # 读取 YAML 文件
    with open(config_filename, "r") as file:
        config = yaml.safe_load(file)

    folder_path = config['folder_path']
    image_folder = os.path.join(folder_path, config['image_folder'])
    excel_file_path = os.path.join(folder_path, config['excel_file_path'])
    is_debug = config["is_debug"]

    #服务名不要改动
    client_option = ClientOption(
        biz_def='mmu',
        grpc_service_name='mmu-chat-gpt-service',
        grpc_stub_class=MmuChatGptServiceStub,
    )

    client = GrpcClient(client_option)
    #chat(client, 60)
    #emb(client, 60)    
    # 获取文件夹中的所有jpg文件
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]

    # 按photo_id分组
    photo_id_to_images = {}
    for image_file in image_files:
        photo_id = image_file.split('_')[0]
        if photo_id not in photo_id_to_images:
            photo_id_to_images[photo_id] = []
        photo_id_to_images[photo_id].append(os.path.join(image_folder, image_file))
    df = pd.read_excel(excel_file_path, engine='openpyxl')
    # 将 Excel 文件中的数据转换为字典
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
    # 使用 ThreadPoolExecutor 进行多线程处理
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    futures = []
    for photo_id, image_files in photo_id_to_images.items():
        futures.append(rate_limited_submit(executor, process_photo, photo_id, image_files, photo_texts, client, config))
        if is_debug:
            break

    # 等待所有线程完成
    for future in concurrent.futures.as_completed(futures):
        try:
            future.result()
        except Exception as e:
            print(f"处理过程中出现错误: {e}")
