import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

import os
# os.system("pip3 install transformers==4.53; pip3 install torchao==0.10; pip3 install easydict") # 请你在import之前安装
import copy
import sys
from pathlib import Path
import os
import easydict
import shutil
os.environ["nosp"] = 'true'
current_script = Path(__file__).resolve()
# print(f"current_script.parent.parent={current_script.parent}")
# sys.path.append(str(current_script.parent))
# pip3 install transformers==4.53; pip3 install torchao==0.10 
from recovlm.models.tokenizer_end2end_mt_1drope_v2.configuration_keye import KeyeConfig
from recovlm.models.tokenizer_end2end_mt_1drope_v2.modeling_keye import KeyeForConditionalGeneration,KeyeImageTokenizer
from recovlm.models.tokenizer_end2end_mt_1drope_v2.keye_vl_utils import process_vision_info

# from recovlm.models.tokenizer_end2end_mt_1drope_v2.modeling_keye_inf2 import KeyeForConditionalGeneration as KeyeForConditionalGeneration_base

from PIL import Image, ImageDraw
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig
import random
import torch
import json
from torch import nn

device = 0

from safetensors.torch import load_file
import tqdm
sd = {}
path = '/mmu_mllm_hdd_2/zhouyang12/output/Keye/vq_end2end_1105/run_exp1.6.6109_stage3/step9500/global_step9500/converted/'
for f in tqdm.tqdm(os.listdir(path)):
    if f.endswith(".safetensors"):
        sd.update(load_file(os.path.join(path, f)))


def generate_circle_image(size=(100, 100), fill_color=(0, 0, 0), outline_color=(255, 255, 255), outline_width=5):
    """
    生成一个包含一个圆的 PIL Image 对象，用于测试。
    
    :param size: 图像的大小，默认为 (64, 64)
    :param fill_color: 圆的填充颜色，默认为黑色 (0, 0, 0)
    :param outline_color: 圆的轮廓颜色，默认为白色 (255, 255, 255)
    :param outline_width: 圆的轮廓宽度，默认为 5
    :return: 生成的 PIL Image 对象
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


model_dir = "/mmu_mllm_hdd_2/zhouyang12/output/Keye/vq_end2end_1105/run_exp1.6.6109_stage3/step9500/global_step9500/converted/"

model = KeyeForConditionalGeneration.from_pretrained(
    model_dir, 
    _attn_implementation="flash_attention_2", 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True
)

model = KeyeForConditionalGeneration(model.config)
model._attn_implementation = "flash_attention_2"
# model.config.output_one_token = model.output_one_token = True
# model.token_head.use_flash_attn = True
model = model.to(device).bfloat16()

model.load_state_dict(sd, strict=True)

# Processor加载和原始输入构建（无Pad）
processor = AutoProcessor.from_pretrained(
    model_dir, 
    trust_remote_code=True
)
## infer image tokens
def process_message(messages):
    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True  # 开启生成提示
    )

    print(f"text={text}")

    image_inputs, video_inputs = process_vision_info(messages)

    # 构建原始输入（纯有效Token，无任何Pad）
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=False,  # 强制关闭Pad，确保原始输入无多余Token
        truncation=False,
        return_tensors="pt",
    ).to(device)
    return inputs


message = [{
        "role": "user",
        "content": [
            {"type": "image", "image": generate_circle_image()},
        ],
    }]
inputs = process_message(
    message
)
image_tokens = model.forward_image_tokens(**inputs)

print(f"image_message={message}")
print(f"image_tokens={image_tokens.shape}")
print(image_tokens)


from recovlm.utils.ds_utils import print_input_info

print_input_info(
    {
        "inputs": inputs,
        "output": image_tokens
    },
    save_path="/mmu_mllm_hdd_2/zhouyang12/output/Keye/vq_end2end_1105/run_exp1.6.6109_stage3/step9500/global_step9500/converted/debug_new/test_run_outputs.pt"
)

