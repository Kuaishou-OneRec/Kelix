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

def set_prec():
    torch.set_printoptions(
        threshold=float('inf'),
        edgeitems=1000,
        linewidth=200,
        sci_mode=False,
        precision=1)

set_prec()

device = 1
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



def test_forward():
    """测试OmniCorpus"""
    COT_SYSTEM_PROMPT = "You are a helpful assistant."
    messages = [
        {"role": "system",
        "content": [
            {"type": "text", "text": COT_SYSTEM_PROMPT},
        ], },               
        {
        "role": "user",
        "content": [
            {"type": "text", "text": " What's sum of the first 10 positive integers? After necessary analysis, your final output should follow the format: Final Answer: X."},
        ],
    }]
    inputs = process_message(messages)
    logits = model(**inputs)
    print(f"logits=\n{logits}")

test_forward()
