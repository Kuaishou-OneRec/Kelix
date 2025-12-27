"""
Qwen3模型前向计算的demo，严格参考tests/test_qwen3.py的参数加载逻辑
"""

import os
from typing import Any, Dict, Optional, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from torch.nn import functional as F
from muse.config import KeyeARConfig
from keye_vl_utils import process_vision_info
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


def process_message(processor, device, messages, add_generation_prompt=True, padding=False):
    """
    处理消息并返回模型输入
    
    Args:
        messages: 消息列表
        add_generation_prompt: 是否添加生成提示
        padding: 是否进行填充
        
    Returns:
        处理后的模型输入
    """
    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=add_generation_prompt
    )
    
    # print(f"text={text}")
    
    image_inputs, video_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=padding,
        truncation=False,
        return_tensors="pt",
    ).to(device)
    return inputs

def generate_and_understanding(model, processor):
    print("Testing generate and understanding...")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Generate an image of a cat."}
            ],
        }
    ]
    inputs = process_message(
        processor, next(model.parameters()).device, messages)
    output_ids = model.generate(**inputs.to(next(model.parameters()).device), top_k=1, max_new_tokens=450)
    output_ids = output_ids[0,inputs["input_ids"].shape[1]:]
    content = processor.decode(output_ids[:,0].long().tolist())
    print(f"输入:\n{messages}")
    print(f"生成内容: {content}\n")
    print(f"output_ids=\n{output_ids}")
    print()
    input_image_ids = model.extract_image_tokens(output_ids)
    print(f"input_image_ids({[x.shape for x in input_image_ids]})=\n{input_image_ids}")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "<|vision_start|><|vision_end|>what's in the image? I need a detailed description."}
            ],
        }
    ]
    inputs = process_message(processor, next(model.parameters()).device, messages)
    inputs["input_ids"] = model.fill_image_tokens(inputs["input_ids"], input_image_ids)
    inputs["input_image_ids"] = torch.cat(input_image_ids, 0)
    inputs = inputs.to(next(model.parameters()).device)
    output_ids = model.generate(**inputs, top_k=1, max_new_tokens=450)
    output_ids = output_ids[0,inputs["input_ids"].shape[1]:]
    content = processor.decode(output_ids[:,0].long().tolist())
    print(f"输入:\n{messages}")
    print(f"生成内容: {content}\n")
    print(f"output_ids=\n{output_ids}")


def edit_and_understanding(model, processor):
    """测试编辑和理解"""
    print("Testing edit and understanding...")
    for size in [300]:
        print(f"Generating circle image of size {size}x{size}...")
        # 生成圆形测试图像
        image = generate_circle_image((size,size))

        # 构建第一次查询：图像内容描述请求
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "What's in the image? I need detailed description."}
                ],
            }
        ]
        # 处理消息并生成多模态输出
        inputs = process_message(processor, next(model.parameters()).device, messages)
        output_ids = model.generate(**inputs, top_k=1, max_new_tokens=400)

        # 解码输出token为文本内容
        content = processor.decode(output_ids[0,inputs["input_ids"].shape[1]:,0].long().tolist())
        print(f"输入:\n{messages}")
        # print(f"生成内容: {content}\n")
        print()

        # 构建第二次查询：图像编辑指令（将图像颜色变为红色）
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Turn the image color into red, but keep the background the same."}
                ],
            }
        ]
        # 处理编辑指令并生成输出
        inputs = process_message(processor, next(model.parameters()).device, messages)
        output_ids = model.generate(**inputs, top_k=1, max_new_tokens=450)
        output_ids = output_ids[0,inputs["input_ids"].shape[1]:]
        content = processor.decode(output_ids[:,0].long().tolist())
        print(f"输入:\n{messages}")
        print(f"生成内容: {content}\n")
        print(f"output_ids=\n{output_ids}")
        print()

        # 从输出中提取图像token用于后续分析
        input_image_ids = model.extract_image_tokens(output_ids)
        print(f"input_image_ids({[x.shape for x in input_image_ids]})=\n{input_image_ids}")
        # 构建第三次查询：纯文本查询（使用特殊token标记图像位置）
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "<|vision_start|><|vision_end|>what's in the image? I need a detailed description."}
                ],
            }
        ]
        # 处理纯文本查询并生成输出
        inputs = process_message(processor, next(model.parameters()).device, messages)
        inputs["input_ids"] = model.fill_image_tokens(inputs["input_ids"], input_image_ids)
        inputs["input_image_ids"] = torch.cat(input_image_ids, 0)
        output_ids = model.generate(**inputs.to(next(model.parameters()).device), top_k=1, max_new_tokens=400)
        output_ids = output_ids[0,inputs["input_ids"].shape[1]:]
        content = processor.decode(output_ids[:,0].long().tolist())
        print(f"setting: size={size}x{size}")
        print(f"输入:\n{messages}")
        print(f"生成内容: {content}\n")


def general_chatting(model, processor):
    device = next(model.parameters()).device
    # 准备输入文本
    for messages in[
            [
                {"role": "user", "content": "Give me a short introduction to large language model."}
            ],
            [
                {"role": "user", "content": [
                    {"type": "image", "image": generate_circle_image()}, 
                    {"type": "text", "text": "What's in the image?"}
                ]}
            ],
            [{"role": "user", "content": "Generate an image of cat."}]
        ]:
        print(f"\n\n\nmessages=\n{messages}")
        # 应用chat template并编码
        inputs = process_message(processor, device, messages)
        print(f"inputs={inputs}")
            
        # 调用generate函数生成文本
        print("\n" + "=" * 60)
        print("使用Muse模型生成文本")
        print("=" * 60)
        
        # 设置生成参数
        generate_params = {
            "max_new_tokens": 450,
            "top_k": 1,
        }
        

        # 生成文本
        print(f"生成参数: {generate_params}")
        print("开始生成...")
        
        generated_ids = model.generate(
            **inputs, 
            **generate_params
        )
        print(f"generated_ids: {generated_ids}")
        # assert torch.all(torch.tensor(generated_ids).to(device) == torch.tensor(outputs).to(device))

        # 解码生成的文本
        generated_text = processor.decode(generated_ids[0,...,0], skip_special_tokens=True)
        
        print("\n生成结果:")
        print(generated_text)


def test_keyear_forward():
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
    processor = AutoProcessor.from_pretrained(
            checkpoint_dir, 
            trust_remote_code=True
        )

    # 创建Muse模型实例
    model_dtype = torch.bfloat16
    with set_default_dtype(model_dtype):
        muse_model = KeyeARModel.from_pretrained(checkpoint_dir).to(device)
    
    general_chatting(muse_model, processor)
    generate_and_understanding(muse_model, processor)
    edit_and_understanding(muse_model, processor)



if __name__ == "__main__":
    test_keyear_forward()