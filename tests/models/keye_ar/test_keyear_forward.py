"""
Qwen3模型前向计算的demo，严格参考tests/test_qwen3.py的参数加载逻辑
"""

import os
from typing import Any, Dict, Optional, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn import functional as F
from muse.config import KeyeARConfig
from muse.models.keye_ar import KeyeARModel
from muse.training.common import set_default_dtype
from PIL import Image, ImageDraw

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


def demo_keyear_forward():
    """
    KeyeAR模型前向计算的demo，严格参考tests/test_keyear.py的实现
    """

    # 设置随机种子
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    device = "cuda:0"
    # 检查预训练模型路径是否存在
    checkpoint_dir = "/mmu_mllm_hdd_2/zhouyang12/output/Keye/vqar_11.7/run_8b_vis_stage3.29_1e-4/step18000/global_step18000/muse_converted"

    # 加载Hugging Face模型和tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)

    # 创建Muse模型实例
    model_dtype = torch.bfloat16
    with set_default_dtype(model_dtype):
        muse_model = KeyeARModel.from_pretrained(checkpoint_dir).to(device)

    # 准备输入文本
    for messages in[
            [
                {"role": "user", "content": "Give me a short introduction to large language model."}
            ],
            [
                {"role": "user", "content": [{
                    "type": "image",
                    "image": generate_circle_image()
                }, 
                {"type": "text", "content": "What's in the image?"}]}
            ],
            [{"role": "user", "content": "Generate an image of cat."}]
        ]:
        print(f"\n\n\nmessages=\n{messages}")
        # 应用chat template并编码
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        print(f"model_inputs={model_inputs}")
        
            
        # 调用generate函数生成文本
        print("\n" + "=" * 60)
        print("使用Muse模型生成文本")
        print("=" * 60)
        
        # 设置生成参数
        generate_params = {
            "max_new_tokens": 20,
            "temperature": 0.8,
            "top_k": 1,
            "top_p": 0.95,
            "eos_token_id": tokenizer.eos_token_id
        }
        
        print(f"Qwen3 baseline generation:")

        # 生成文本
        print(f"生成参数: {generate_params}")
        print("开始生成...")
        
        generated_ids = muse_model.generate(
            model_inputs["input_ids"], 
            **generate_params
        )
        print(f"generated_ids: {generated_ids}")
        # assert torch.all(torch.tensor(generated_ids).to(device) == torch.tensor(outputs).to(device))

        # 解码生成的文本
        generated_text = tokenizer.decode(generated_ids[0,...,0], skip_special_tokens=True)
        
        print("\n生成结果:")
        print(generated_text)
        



if __name__ == "__main__":
    demo_keyear_forward()