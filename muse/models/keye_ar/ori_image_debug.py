"""
多模态模型测试框架
支持添加多个测试用例并批量运行
"""

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["nosp"] = 'true'
from tqdm import tqdm
import IPython
import sys
from pathlib import Path
import torch
import re
import time
from PIL import Image, ImageDraw
from transformers import AutoProcessor

# 设置环境和路径
current_script = Path(__file__).resolve()



# muse/models/keye_ar/ori_image_debug.py
# 导入模型相关模块
# muse/models/keye_ar/keye_vl_utils.py
from muse.models.keye_ar.ori_image_debug import KeyeForConditionalGeneration
from muse.models.keye_ar.keye_vl_utils import process_vision_info


def set_prec():
    torch.set_printoptions(
        threshold=float('inf'),
        edgeitems=1000,
        linewidth=200,
        sci_mode=False,
        precision=1)

set_prec()

device = 0 
output_model_dir = "/mmu_mllm_hdd_2/zhouyang12/output/Keye/vqar_11.7/run_8b_vis_stage3.29_1e-4/step12000/global_step12000/converted"

model = KeyeForConditionalGeneration.from_pretrained(
            output_model_dir, 
            _attn_implementation="flash_attention_2", 
            torch_dtype=torch.bfloat16, 
            low_cpu_mem_usage=True
        )
model.config.output_one_token = model.output_one_token = False
model.token_head.use_flash_attn = True
model = model.to(device).bfloat16()
processor = AutoProcessor.from_pretrained(
            output_model_dir, 
            trust_remote_code=True
        )
        

def process_message( messages, add_generation_prompt=True, padding=False):
    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=add_generation_prompt
    )
    
    # print(f"text={text}")
    
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=padding,
        truncation=False,
        return_tensors="pt",
    ).to(device)
    return inputs


def generate_circle_image(size=(100, 100), fill_color=(0, 0, 0), outline_color=(255, 255, 255), outline_width=5):
    """
    生成一个包含一个圆的 PIL Image 对象，用于测试。
    
    Args:
        size: 图像的大小，默认为 (100, 100)
        fill_color: 圆的填充颜色，默认为黑色 (0, 0, 0)
        outline_color: 圆的轮廓颜色，默认为白色 (255, 255, 255)
        outline_width: 圆的轮廓宽度，默认为 5
        
    Returns:
        生成的 PIL Image 对象
    """
    # 创建一个新的图像对象
    image = Image.new('RGB', size, color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    # 计算圆的坐标（图像中心为圆心）
    x_center, y_center = size[0] // 2, size[1] // 2
    radius = min(size[0], size[1]) // 2
    # 绘制圆
    draw.ellipse([x_center - radius, y_center - radius, x_center + radius, y_center + radius],
                 fill=fill_color,
                 outline=outline_color,
                 width=outline_width)
    return image


def test_image_generation_cust():
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": generate_circle_image()}
            {"type": "text", "text": " What's in the image?"},
        ]
    }]
    inputs = process_message(messages)
    outputs = model(**inputs)
    for k,v in outputs.items():
        try:
            print(f"test_image_generation_cust-{k}: {v.shape}")
        except:
            print(f"test_image_generation_cust-{k}: {v}")



test_image_generation_cust()
