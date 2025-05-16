from transformers import AutoModelForCausalLM, AutoTokenizer
# from recovlm.models.qwen_3_vl.modeling_qwen3_vl import Qwen3_VLForConditionalGeneration_siglip
# from recovlm.models.qwen_3_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor_siglip
from typing import Dict, Any, Union, Optional
from recovlm.utils.ds_utils import format_dict_or_list
from transformers import AutoTokenizer, AutoProcessor

import contextlib
import gc
import argparse
import time
import datetime
import os
import glob
import json
import logging
import collections
import pickle
import itertools
import contextlib
import multiprocessing as mp
from functools import partial

from PIL import Image, ImageDraw

from recovlm.models.qwen_2_5_vl.checkpoint import Qwen2_5_VL_siglipCheckpointConverter


from recovlm.utils.ds_utils import print_input_info

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np

from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from transformers import AutoTokenizer

import os
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from transformers import GenerationConfig
from recovlm.training.common import set_default_dtype, get_global_grad_norm, clip_grad_by_value
from functools import partial
from recovlm.utils.common import set_random_seed, to_cuda, print_rank_0
from recovlm.training.distributed import shard_model, get_shard_conditions, \
  load_from_full_model_state_dict
from recovlm.training.checkpoint import load_hf_checkpoint
import itertools
from recovlm.utils.ds_utils import print_input_info
from recovlm.utils.qwen_vl_utils import process_vision_info

from recovlm.training.parallel import initialize_model_parallel
from recovlm.models.qwen2_vl.checkpoint import Qwen2VLCheckpointConverter

from typing import Dict, Any, Union, Optional

import contextlib
import gc
import argparse
import time
import datetime
import os
import glob
import json
import logging
import collections
import pickle
import itertools
import contextlib
from functools import partial


from recovlm.utils.ds_utils import print_input_info
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np


from recovlm.data.dataloaders_v2 import get_dataloader as get_dataloader_v2


from recovlm.training.common import set_default_dtype, get_global_grad_norm, clip_grad_by_value

from recovlm.models.internvl.modeling_internvl_chat import InternVLChatModel
from recovlm.utils.time_tracker import TimeTracker
from recipes.inspects import info_params_recursive

# /llm_reco/lingzhixin/recovlm_qw0510/recovlm/recovlm/models/qwen3siglip/processing_qwen3siglip.py
from recovlm.models.qwen3siglip.processing_qwen3siglip import Qwen3SiglipProcessor_navit

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


import math
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images




# # initialize distributed environment
# rank = int(os.environ["RANK"])
# world_size = int(os.environ.get("WORLD_SIZE", "1"))
# local_rank = int(os.environ.get("LOCAL_RANK", "0"))
# device = torch.device(f"cuda:{local_rank}")
torch.cuda.set_device(0)
# torch.cuda.set_default_device(0)

# torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)

# rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", 0))
# world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", 0))
# local_rank = rank % world_size
local_rank = 0
# print_rank_0(rank, world_size, local_rank)
# torch init
torch.cuda.set_device(local_rank)
# torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)


MODEL_DIR="/llm_reco_ssd/zhouyang12/models/msy_Qwen3vl-8B-Base"
MODEL_DIR="/llm_reco_ssd/zhouyang12/models/Qwen3-8B-Base-siglip"
MODEL_DIR="/llm_reco_ssd/zhouyang12/models/Qwen3-8B-Base-siglip"


from recovlm.models.internvl.checkpoint import InternVLCheckpointConverter
# MODEL_DIR="/llm_reco/lingzhixin/output2/RecoVLM-dev/Qwen2-VL-7B-run_sft_7B_fsdp_sp/0.0.5/_1000/global_step_1000_torch_ckpt/"

# Qwen2_5_VLForConditionalGeneration_siglip


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


def load_image(image_file=None, input_size=448, max_num=12):
    # image = Image.open(image_file).convert('RGB')
    image = generate_circle_image()
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

processor = AutoProcessor.from_pretrained('/llm_reco_ssd/zhouyang12/models/InternVL3-2B')
converter = InternVLCheckpointConverter('/llm_reco_ssd/zhouyang12/models/InternVL3-2B')


tokenizer = AutoTokenizer.from_pretrained('/llm_reco_ssd/zhouyang12/models/InternVL3-2B')


def make_inputs(a,b):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": generate_circle_image((a,b),) },
                {"type": "text", "text": "what's in the image"},
            ],
        }
    ]

    # /llm_reco_ssd/zhouyang12/models/Qwen3-1.7B-siglip2/config.json

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
    return messages, inputs


from transformers import AutoTokenizer, AutoModel
generation_config = dict(max_new_tokens=1024, do_sample=True)

logits_all = []
if 1:
    try:
        with set_default_dtype(torch.bfloat16):
            model = AutoModel.from_pretrained(
                '/llm_reco_ssd/zhouyang12/models/InternVL3-2B',
                torch_dtype=torch.bfloat16,
                load_in_8bit=True,
                low_cpu_mem_usage=True,
                use_flash_attn=True,
                trust_remote_code=True
            )
            # model.load_state_dict(state_dict)

            question = '<image>\nPlease describe the image in detail.'
            response, history = model.chat(tokenizer, load_image(), question, generation_config, history=None, return_history=True)
            print(f'User: {question}\nAssistant: {response}')


    except Exception as e:
        import traceback
        traceback.print_exc()
        print(e)
        pass
