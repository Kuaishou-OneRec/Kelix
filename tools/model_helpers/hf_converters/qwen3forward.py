import torch 
import torch.nn as nn


def num_params(model):
    return sum([x.numel() for x in model.parameters()])


def info_params_recursive(model, name="", max_depth=5, curr_depth=0):
    """
    from torchvision import models
    print(info_params_recursive(models.resnet18()))
    """
    res = ""
    if curr_depth == 0:
        res += "下面每行的格式为:\n当前深度-<模型类型>(模型名称): 参数数量\t\tp0:第一个参数名:第一个参数均值\n"
    if curr_depth == max_depth: return ""
    #
    indent = '--' * (curr_depth + 1)
    named_params = list(model.named_parameters())
    if len(named_params):
        pname, pparam = sorted(named_params)[0]
        pparam = pparam.detach().mean().item()
    else:
        pname, pparam = None, None
    res += "{} {}-{}({}): {}\t\tp0:{}:{}\n".format(indent, curr_depth, type(model), name, num_params(model), pname, pparam)
    for name, model in model.named_children():
        if isinstance(model, nn.Module):
            res += info_params_recursive(model, name, max_depth, curr_depth + 1)
    return res

from PIL import Image, ImageDraw
from PIL import Image
import torch
import sys
from transformers import AutoTokenizer, AutoModel, AutoProcessor
import contextlib
from transformers import AutoTokenizer, AutoModel, AutoProcessor, AutoModelForCausalLM
import os
import shutil
import json



torch.cuda.set_device(0)
torch.set_default_dtype(torch.bfloat16)

current_script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(current_script_path)

MODEL_DIR = "/mmu_mllm_hdd_2/lingzhixin/models/Keye-32B-scratch_0606" if len(sys.argv) == 1 else sys.argv[1]
# print(f"MODEL_DIR={MODEL_DIR}")

# def copy_to_model_dir(model_dir):
#     print(f"copying {script_dir} -> {model_dir}")
#     for fn in os.listdir(script_dir):
#         if fn == '__pycache__': continue
#         src = os.path.join(script_dir, fn)
#         dst = os.path.join(model_dir, fn)
#         overwrite = os.path.exists(dst)
#         shutil.copy(src, dst)
#         print(f"copying {src} -> {dst}", "overwrite" if overwrite else "")

#     print(f"converting is done... check {model_dir}")

# copy_to_model_dir(MODEL_DIR)


from keye_vl_utils import process_vision_info
def set_seed(seed: int):
    import random
    import numpy as np

    """设置所有可能的随机数种子，保证实验可重复性"""
    # 设置 Python 内置的随机数种子
    random.seed(seed)
    # 设置 NumPy 的随机数种子
    np.random.seed(seed)
    # 设置 PyTorch 的 CPU 随机数种子
    torch.manual_seed(seed)
    # 设置 PyTorch 的 CUDA 随机数种子（用于 GPU 计算）
    torch.cuda.manual_seed(seed)
    # 如果使用了多个 GPU，还需要设置这个
    torch.cuda.manual_seed_all(seed)
    # 禁用 CuDNN 的非确定性算法（确保结果可复现）
    torch.backends.cudnn.deterministic = True
    # 禁用 CuDNN 的自动调优功能（确保每次运行使用相同的算法）
    torch.backends.cudnn.benchmark = False

set_seed(0)
local_rank = 0


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
    



def generate_circle_image(size=(200, 200), fill_color=(0, 0, 0), outline_color=(255, 255, 255), outline_width=5):
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


processor = AutoProcessor.from_pretrained("/mmu_mllm_hdd_2/lingzhixin/models/Qwen3-32B", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("/mmu_mllm_hdd_2/lingzhixin/models/Qwen3-32B", trust_remote_code=True)

def make_inputs(a,b, _type="image"):
    mm = [{"type": "image", "image": generate_circle_image((a,b),) }] if _type == "image" else [{"type": "video", "video": "/llm_reco/lingzhixin/recovlm_data/tests/2.mp4"}]
    if _type is None: mm = [{"type": "text", "text": ""}]
    if _type == "both":
        mm = [
            {"type": "video", "video": "/llm_reco/lingzhixin/recovlm_data/tests/2.mp4", "max_pixels": 64*32*32}, # 这里是demo, 评估可以用更大的max_pixels
            {"type": "image", "image": generate_circle_image((a,b),) },
        ]

    messages = [
        {
            "role": "user",
            "content": [
                "what's in the image"
                # *mm,
                # {"type": "text", "text": "what's in the image"},
            ],
        }
    ]
    
    import copy
    messages_ = copy.deepcopy(messages)
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    # image_inputs, video_inputs = process_vision_info(messages)
    print(messages)
    print(text)
    inputs = tokenizer(
        text=[text],
        #images=image_inputs,
        #videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(local_rank)
    return messages, inputs, messages_

# make_inputs(100,100); exit()

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16,
    _attn_implementation = 'flash_attention_2',
    device_map="cuda:0",
    ignore_mismatched_sizes=True,
    trust_remote_code=True
)
# print(info_params_recursive( model, max_depth=5)); exit()


def generate_ascii_banner(text, width=None, padding=50, char='='):
    if width is None:
        width = len(text) + 2 * padding
    
    min_width = len(text) + 4
    if width < min_width:
        width = min_width
    
    total_decor = width - len(text)
    left_decor = total_decor // 2
    right_decor = total_decor - left_decor
    
    decoration = char * width
    content_line = f"{char * left_decor}{text}{char * right_decor}"
    
    return f"{decoration}\n{content_line}\n{decoration}"
    
for _type in [ None]:
    print(generate_ascii_banner("testing " + str(_type) +  "..." ))
    messages, inputs, messages_old = make_inputs(100,100, _type)
    generated = model.generate(**inputs,     top_k=1, max_new_tokens=256) 
    logits = model(**inputs).logits
    output_ids = generated[0][len(inputs.input_ids[0]):].tolist() 
    content = tokenizer.decode(output_ids[0:], skip_special_tokens=True).strip("\n")
    messages = messages[0]

    messages["input_ids"] = inputs.input_ids.shape
    messages["pixel_values"] = inputs.pixel_values.shape if hasattr(inputs, "pixel_values") else None
    messages["pixel_values_videos"] = inputs.pixel_values_videos.shape if hasattr(inputs, "pixel_values_videos") else None

    messages["messages"] = messages_old
    messages["content"] = content
    messages["logits"] = logits # [-1, -3:]
    messages["logits.shape"] = logits.shape
    print(format_dict_or_list(messages))


"""
{
  "role": "user",
  "content": "It seems like your message might not have come through as intended. Could you please clarify or provide more details about what you're asking? I'm here to help!",
  "input_ids": torch.Size([1, 5]),
  "pixel_values": None,
  "pixel_values_videos": None,
  "messages": [
    {
      "role": "user",
      "content": [
        "what's in the image"
      ]
    }
  ],
  "logits": tensor([[[ 4.8438,  3.0781,  3.0000,  ..., -2.5625, -2.5625, -2.5625],
         [-0.2871,  4.4062,  1.3906,  ..., -7.1875, -7.1562, -7.3125],
         [ 1.0469,  6.9062,  4.5938,  ..., -7.6562, -7.6562, -7.8750],
         [ 3.0938,  2.5312,  2.9844,  ..., -2.3750, -2.3750, -2.4531],
         [ 4.3125,  5.4062, 12.3125,  ...,  3.1719,  3.1719,  3.1250]]],
       device='cuda:0', grad_fn=<UnsafeViewBackward0>),
  "logits.shape": torch.Size([1, 5, 151936])
}
"""