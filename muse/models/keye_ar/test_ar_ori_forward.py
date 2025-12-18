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
import argparse  # 添加argparse模块导入


# 导入模型相关模块
# from recovlm.models.tokenizer_end2end_mt_1drope_v8.configuration_keye import KeyeConfig

from muse.models.keye_ar.ar_ori import KeyeForConditionalGeneration#, KeyeImageTokenizer

from keye_vl_utils import process_vision_info
import IPython


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

def set_prec():
    torch.set_printoptions(
        threshold=float('inf'),
        edgeitems=1000,
        linewidth=200,
        sci_mode=False,
        precision=1)

# set_prec()

# 使用明确的 device，并在可能时使用 cuda:1
device = 0 # torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
output_model_dir = "/mmu_mllm_hdd_2/zhouyang12/output/Keye/vqar_11.7/run_8b_vis_stage3.2/step5000/global_step5000/converted/"
output_model_dir = "/mmu_mllm_hdd_2/zhouyang12/output/Keye/vqar_11.7/run_8b_vis_stage3.29_1e-4/step4000/global_step4000/converted"

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
IPython.embed()

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

    # 将 inputs 中所有浮点类型的 Tensor 转为 bfloat16
    def _cast_inputs_to_bf16(batch):
        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                batch[k] = v.to(dtype=torch.bfloat16)
        return batch

    inputs = _cast_inputs_to_bf16(inputs)
    # 确保 inputs 全部在目标 device 上
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}


def test_forward():
    """测试OmniCorpus"""
    COT_SYSTEM_PROMPT = "You are a helpful assistant."
    COT_SYSTEM_PROMPT = "You are a helpful assistant."
    messages = [
        {"role": "system",
         "content": [
             {"type": "text", "text": COT_SYSTEM_PROMPT},
         ], },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": generate_circle_image()},
                {"type": "text", "text": " What's sum of the first 10 positive integers? After necessary analysis, your final output should follow the format: Final Answer: X."},
            ],
        }
    ]
    
    inputs = process_message(messages)#.to(device)

    # 在 forward 时使用 autocast 来强制内部 float ops 使用 bfloat16，避免 float/bfloat16 混合导致 upcast
    if torch.cuda.is_available():
        autocast_cm = torch.cuda.amp.autocast
    else:
        # CPU 上也可以使用 bfloat16 autocast（需要对应 PyTorch 版本）
        try:
            autocast_cm = torch.cpu.amp.autocast
        except Exception:
            autocast_cm = nullcontext  # fallback，若没有 cpu autocast 则不使用

    from contextlib import nullcontext
    with autocast_cm(dtype=torch.bfloat16):
        logits = model(**inputs)
    print(f"logits=\n{logits}")

test_forward()
