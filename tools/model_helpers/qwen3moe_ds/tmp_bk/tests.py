from PIL import Image, ImageDraw
from PIL import Image
import torch
import sys
from transformers import AutoTokenizer, AutoModel, AutoProcessor
import contextlib
from transformers import AutoTokenizer, AutoModel, AutoProcessor
import os
import shutil
import json



torch.cuda.set_device(0)
torch.set_default_dtype(torch.bfloat16)

current_script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(current_script_path)

MODEL_DIR = "/llm_reco/lingzhixin/models/Keye-8B-demo_dev" if len(sys.argv) == 1 else sys.argv[1]
print(f"MODEL_DIR={MODEL_DIR}")

def copy_to_model_dir(model_dir):
    print(f"copying {script_dir} -> {model_dir}")
    for fn in os.listdir(script_dir):
        if fn == '__pycache__': continue
        src = os.path.join(script_dir, fn)
        dst = os.path.join(model_dir, fn)
        overwrite = os.path.exists(dst)
        shutil.copy(src, dst)
        print(f"copying {src} -> {dst}", "overwrite" if overwrite else "")

    print(f"converting is done... check {model_dir}")

copy_to_model_dir(MODEL_DIR)


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


processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
tokenizer = processor.tokenizer

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
                *mm,
                {"type": "text", "text": "what's in the image"},
            ],
        }
    ]
    
    import copy
    messages_ = copy.deepcopy(messages)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(local_rank)
    return messages, inputs, messages_

model = AutoModel.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16,
    _attn_implementation = 'flash_attention_2',
    device_map="cuda:0",
    # ignore_mismatched_sizes=True,
    trust_remote_code=True
)

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
    
for _type in ["image", "video", "both", None]:
    print(generate_ascii_banner("testing " + str(_type) +  "..." ))
    messages, inputs, messages_old = make_inputs(100,100, _type)
    generated = model.generate(**inputs, max_new_tokens=256) 
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
预期输出
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 27/27 [00:19<00:00,  1.42it/s]
====================================================================================================================
==================================================testing image...==================================================
====================================================================================================================
{
  "role": "user",
  "content": "The image depicts a close-up view of a black, reflective surface. The texture of the surface appears smooth and glossy, reflecting light in a way that creates a mirror-like effect. The reflection is somewhat blurred, suggesting that the surface might be wet or that the light source is not directly above the surface.",
  "input_ids": torch.Size([1, 32]),
  "pixel_values": torch.Size([36, 3, 16, 16]),
  "pixel_values_videos": None,
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "image",
          "image": <PIL.Image.Image image mode=RGB size=100x100 at 0x7F97596BEDD0>
        },
        {
          "type": "text",
          "text": "what's in the image"
        }
      ]
    }
  ],
  "logits": tensor([[[ 6.0312,  5.5000,  4.1562,  ..., -0.7578, -0.7578, -0.7578],
         [-1.1328,  2.5625,  4.8125,  ..., -3.0625, -3.0469, -3.0625],
         [-3.8906,  0.0645,  6.4062,  ..., -3.5625, -3.5625, -3.5625],
         ...,
         [10.6250,  9.9375,  8.6250,  ..., -2.0156, -2.0156, -2.0156],
         [ 0.2637,  0.4395, -3.4688,  ...,  1.9922,  1.9922,  1.9922],
         [ 5.2500, 11.3125,  5.8750,  ..., -2.0781, -2.0781, -2.0781]]],
       device='cuda:0', grad_fn=<UnsafeViewBackward0>),
  "logits.shape": torch.Size([1, 32, 151936])
}
====================================================================================================================
==================================================testing video...==================================================
====================================================================================================================
qwen-vl-utils using decord to read video.
{
  "role": "user",
  "content": "在《大明王朝1566》中，朱厚熜的扮演者是刘涛。",
  "input_ids": torch.Size([1, 4343]),
  "pixel_values": None,
  "pixel_values_videos": torch.Size([17280, 3, 16, 16]),
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "video",
          "video": "/llm_reco/lingzhixin/recovlm_data/tests/2.mp4"
        },
        {
          "type": "text",
          "text": "what's in the image"
        }
      ]
    }
  ],
  "logits": tensor([[[ 6.3125,  5.7812,  4.4062,  ..., -0.7578, -0.7578, -0.7578],
         [-1.0547,  2.4844,  4.8125,  ..., -2.9688, -2.9688, -2.9688],
         [-3.9219, -0.0217,  6.1875,  ..., -3.5625, -3.5625, -3.5625],
         ...,
         [10.8750,  9.0000,  8.6875,  ..., -1.0469, -1.0469, -1.0469],
         [-0.7305, -5.0000, -4.3750,  ..., -0.4238, -0.4238, -0.4238],
         [ 4.1562,  6.6250,  7.4375,  ..., -2.2188, -2.2188, -2.2188]]],
       device='cuda:0', grad_fn=<UnsafeViewBackward0>),
  "logits.shape": torch.Size([1, 4343, 151936])
}
===================================================================================================================
==================================================testing both...==================================================
===================================================================================================================
{
  "role": "user",
  "content": "The image shows a scene from a movie. In the foreground, there is a man wearing a black suit and holding a sword. He is standing in front of a large, ornate building with many columns. The building has a golden dome on top. The man is looking at the camera, and there is a woman in the background, partially visible. She is wearing a white dress and is also standing in front of the building. The setting appears to be a palace or a similar grandiose structure, and the scene suggests a dramatic or historical context.",
  "input_ids": torch.Size([1, 394]),
  "pixel_values": torch.Size([36, 3, 16, 16]),
  "pixel_values_videos": torch.Size([1440, 3, 16, 16]),
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "video",
          "video": "/llm_reco/lingzhixin/recovlm_data/tests/2.mp4",
          "max_pixels": 65536
        },
        {
          "type": "image",
          "image": <PIL.Image.Image image mode=RGB size=100x100 at 0x7F975937FAF0>
        },
        {
          "type": "text",
          "text": "what's in the image"
        }
      ]
    }
  ],
  "logits": tensor([[[ 6.2188,  5.7188,  4.3438,  ..., -0.7617, -0.7617, -0.7617],
         [-1.1016,  2.5469,  4.7812,  ..., -3.0312, -3.0312, -3.0312],
         [-3.7969,  0.1592,  6.4688,  ..., -3.5469, -3.5469, -3.5469],
         ...,
         [10.0000,  8.5625,  8.8750,  ..., -1.3828, -1.3828, -1.3828],
         [ 5.1562,  3.3281, -0.2930,  ..., -2.2188, -2.2188, -2.2188],
         [ 6.0625,  9.1875,  7.6250,  ..., -1.6953, -1.6953, -1.6953]]],
       device='cuda:0', grad_fn=<UnsafeViewBackward0>),
  "logits.shape": torch.Size([1, 394, 151936])
}
===================================================================================================================
==================================================testing None...==================================================
===================================================================================================================
{
  "role": "user",
  "content": "The image features a vibrant and colorful collage of various objects and scenes. Here's a detailed description of what can be observed:

1. **Tree Trunk**: A large, green tree trunk is prominently displayed in the center of the image. The trunk has a rough texture and appears to be a significant part of the composition.

2. **Flowers**: There are several flowers scattered around the tree trunk. These flowers come in a variety of colors, including pink, yellow, and white. They add a touch of natural beauty to the image.

3. **Leaves**: Numerous leaves are present, adding to the lush and green appearance of the tree. The leaves are a mix of green and yellow, suggesting the presence of autumn foliage.

4. **Birds**: Two birds are visible in the image. One bird is perched on the tree trunk, while the other is flying in the background. The birds add a sense of movement and life to the scene.

5. **Clouds**: A few clouds can be seen in the background, adding a soft and gentle touch to the overall composition.

6. **Text**: The image contains text, which is not clearly legible in the provided description. However, it appears to be in a language that is not English.",
  "input_ids": torch.Size([1, 21]),
  "pixel_values": None,
  "pixel_values_videos": None,
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": ""
        },
        {
          "type": "text",
          "text": "what's in the image"
        }
      ]
    }
  ],
  "logits": tensor([[[ 6.1875,  5.6562,  4.2812,  ..., -0.7617, -0.7617, -0.7617],
         [-1.1094,  2.6094,  4.6875,  ..., -3.0625, -3.0625, -3.0625],
         [-3.7812,  0.2227,  6.4375,  ..., -3.5469, -3.5469, -3.5469],
         ...,
         [10.6250,  9.8750,  9.1250,  ..., -1.6641, -1.6641, -1.6641],
         [ 2.2344,  1.3281, -3.7344,  ..., -1.0391, -1.0391, -1.0391],
         [ 4.9688, 10.8750,  9.5000,  ..., -0.6680, -0.6680, -0.6680]]],
       device='cuda:0', grad_fn=<UnsafeViewBackward0>),
  "logits.shape": torch.Size([1, 21, 151936])
}




版本
root@aiplatform-wlf2-ge27-111:/llm_reco/lingzhixin/pub_models/models/versions# pip3 show transformers
Name: transformers
Version: 4.49.0
Summary: State-of-the-art Machine Learning for JAX, PyTorch and TensorFlow
Home-page: https://github.com/huggingface/transformers
Author: The Hugging Face team (past and future) with the help of all our contributors (https://github.com/huggingface/transformers/graphs/contributors)
Author-email: transformers@huggingface.co
License: Apache 2.0 License
Location: /usr/local/lib/python3.10/dist-packages
Requires: filelock, huggingface-hub, numpy, packaging, pyyaml, regex, requests, safetensors, tokenizers, tqdm
Required-by: compressed-tensors, vllm, xgrammar

root@aiplatform-wlf2-ge27-111:/llm_reco/lingzhixin/pub_models/models/versions# pip3 show torch
Name: torch
Version: 2.5.1+cu118
Summary: Tensors and Dynamic neural networks in Python with strong GPU acceleration
Home-page: https://pytorch.org/
Author: PyTorch Team
Author-email: packages@pytorch.org
License: BSD-3-Clause
Location: /usr/local/lib/python3.10/dist-packages
Requires: filelock, fsspec, jinja2, networkx, nvidia-cublas-cu11, nvidia-cuda-cupti-cu11, nvidia-cuda-nvrtc-cu11, nvidia-cuda-runtime-cu11, nvidia-cudnn-cu11, nvidia-cufft-cu11, nvidia-curand-cu11, nvidia-cusolver-cu11, nvidia-cusparse-cu11, nvidia-nccl-cu11, nvidia-nvtx-cu11, sympy, triton, typing-extensions
Required-by: accelerate, compressed-tensors, deepspeed, flash-attn, outlines, timm, torchaudio, torchdata, torchvision, vllm, xformers, xgrammar
"""