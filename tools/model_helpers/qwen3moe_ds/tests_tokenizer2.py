from PIL import Image, ImageDraw
from PIL import Image
import torch
import sys
from transformers import AutoTokenizer, AutoModel, AutoProcessor
import contextlib
import os
import shutil
import json




MODEL_DIR3 = "DeepSeek-R1-0528-Qwen3-8B/"

MODEL_DIR = MODEL_DIR3
# MODEL_DIR = "/mmu_mllm_hdd_2/lingzhixin/models/Keye-R1-0528-8B-vit-scratch_v0_8_1" if len(sys.argv) == 1 else sys.argv[1]
print(f"MODEL_DIR={MODEL_DIR}")


def format_dict_or_list(obj, indent_level=0, indent_size=2):
    """
    格式化打印dict/list，用来替代json.dumps
    """
    def format_value(value, indent_level=0, indent_size=2):
        if isinstance(value, (dict, list)):
            return format_dict_or_list(value, indent_level, indent_size)
        elif isinstance(value, str):
            return f'"{value}"'
        else:
            return str(value)

    if isinstance(obj, dict):
        items = [f": {format_value(v, indent_level + 1)}" for k, v in obj.items()]
        keys = [f'"{k}"' for k in obj.keys()]
        formatted_items = ',\n'.join(f'{(" " * indent_size * (indent_level + 1))}{k}{v}' for k, v in zip(keys, items))
        return '{\n' + formatted_items + '\n' + (' ' * indent_size * indent_level) + '}'
    elif isinstance(obj, list):
        items = [format_value(item, indent_level + 1) for item in obj]
        formatted_items = ',\n'.join(' ' * indent_size * (indent_level + 1) + item for item in items)
        return '[\n' + formatted_items + '\n' + (' ' * indent_size * indent_level) + ']'
    else:
        return obj
    


def generate_circle_image(size=(50, 50), fill_color=(0, 0, 0), outline_color=(255, 255, 255), outline_width=5):
    """
    生成一个包含一个圆的 PIL Image 对象。

    :param size: 图像的大小，默认为 (200, 200)
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


def make_inputs(a,b, model_dir):
    # https://huggingface.co/datasets/MathLLMs/MathVision, mathvision 把图片放在后面
    # processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    # tokenizer = processor.tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    messages = [
        {
            "role": "user",
            # "content": [
            #     # {"type": "image", "image": generate_circle_image()},
            #     # {"type": "text", "text": "How are you?"},
            # ],
            "content": "hello how are"
        }
    ]
    
    import copy
    messages_ = copy.deepcopy(messages)
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    #image_inputs, video_inputs = process_vision_info(messages)
    inputs = tokenizer(
        text=[text],
        # images=image_inputs,
        # videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return {
        "messages_": messages_,
        "text": text,
        "inputs": inputs
    }


for d in [MODEL_DIR3]:
    print("=" * 20)
    print(d)
    print(format_dict_or_list(
        make_inputs(100,100,d)
    ))

