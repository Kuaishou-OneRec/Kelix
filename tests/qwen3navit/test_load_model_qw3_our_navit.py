from transformers import AutoModelForCausalLM, AutoTokenizer
# from recovlm.models.qwen_3_vl.modeling_qwen3_vl import Qwen3_VLForConditionalGeneration_siglip
# from recovlm.models.qwen_3_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor_siglip
from typing import Dict, Any, Union, Optional
from recovlm.utils.ds_utils import format_dict_or_list

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


from recovlm.training.checkpoint import AppState, DistributedCheckpointer
from recovlm.models.qwen2_vl.checkpoint import Qwen2VLCheckpointConverter
from recovlm.models.internvl.checkpoint import InternVLCheckpointConverter

from recovlm.utils.ds_utils import print_input_info
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np

from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from transformers import AutoTokenizer
from recovlm.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
from recovlm.models.qwen2_vl import Qwen2VLForConditionalGeneration

from recovlm.models.qwen_2_5_vl import Qwen2_5_VLForConditionalGeneration
from recovlm.models.qwen_2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor

from recovlm.models.internvl import InternVLChatModel
from recovlm.models.qwen2 import Qwen2DecoderLayer
from recovlm.models.internvl import InternVisionEncoderLayer

from recovlm.data.dataloaders_v2 import get_dataloader as get_dataloader_v2
from recovlm.data.dataloaders import get_dataloader

from recovlm.utils.merge_checkpoints import convert_zero_checkpoint_to_state_dict
from recovlm.losses import CrossEntropyLoss
from recovlm.utils.common import set_random_seed, to_cuda, print_rank_0, \
  get_optimizer_grouped_parameters, dist_reduce_dict, Timer, heart_beat
from recovlm.training.lr_schedulers import get_scheduler

from recovlm.training.parallel import get_sequence_parallel_group, \
  get_sequence_parallel_rank, get_sequence_parallel_world_size, \
  get_local_sequence_boundary, initialize_model_parallel, gather_by_group, \
  get_local_sequence, get_data_parallel_group, get_data_parallel_world_size, \
  get_data_parallel_rank

from torch.distributed.device_mesh import init_device_mesh, DeviceMesh

from recovlm.training.distributed import shard_model, get_shard_conditions, \
  load_from_full_model_state_dict
from recovlm.training.checkpoint import load_hf_checkpoint

from recovlm.training.activations import set_activation_checkpointing

from recovlm.training.common import set_default_dtype, get_global_grad_norm, clip_grad_by_value

from recovlm.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLDecoderLayer, Qwen2VLVisionBlock
# from recovlm.models.qwen_2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLDecoderLayer, Qwen2_5_VLVisionBlock

# from recovlm.models.qwen_3_vl_2.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration_siglip, Qwen2_5_VLForConditionalGeneration_siglip_navit
# /llm_reco/lingzhixin/recovlm_qw0510/recovlm/recovlm/models/qwen3siglip/modeling_qwen3siglip.py
from recovlm.models.qwen3siglip.modeling_qwen3siglip import Qwen3SiglipForConditionalGeneration_navit
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

# # initialize distributed environment
# rank = int(os.environ["RANK"])
# world_size = int(os.environ.get("WORLD_SIZE", "1"))
# local_rank = int(os.environ.get("LOCAL_RANK", "0"))
# device = torch.device(f"cuda:{local_rank}")
# torch.cuda.set_device(device)
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

# MODEL_DIR="/llm_reco/lingzhixin/output2/RecoVLM-dev/Qwen2-VL-7B-run_sft_7B_fsdp_sp/0.0.5/_1000/global_step_1000_torch_ckpt/"

# Qwen2_5_VLForConditionalGeneration_siglip


model_name = "Qwen/Qwen3-8B"
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


processor = Qwen3SiglipProcessor_navit.from_pretrained('/llm_reco_ssd/zhouyang12/models/Qwen3-1.7B-siglip2')



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
    from recovlm.models.qwen3siglip.processing_qwen3siglip import Qwen3SiglipProcessor_navit

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
    return inputs




# messagest = [
#     {
#         "role": "user",
#         "content": [
#             {"type": "text", "text": "what's in the image"},
#         ],
#     }
# ]
# textt = processor.apply_chat_template(
#     messagest, tokenize=False, add_generation_prompt=False
# )
# inputst = processor(
#     text=[text],
#     padding=True,
#     return_tensors="pt",
# )

# print(inputs)
'''
{'input_ids': tensor([[151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13,
         151645,    198, 151644,    872,    198,   4340,    525,    498, 151645,
            198, 151644,  77091,    198]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}
'''


def load_from_full_model_state_dict(model, full_sd: Dict[str, Any], allow_random_init_params="mlp_AR.pre_norm.weight,mlp_AR.pre_norm.bias,mlp_AR.linear_1.weight,mlp_AR.linear_1.bias,mlp_AR.linear_2.weight,mlp_AR.linear_2.bias"):
    # allow_random_init_params = ['mlp_AR.pre_norm.weight', 'mlp_AR.pre_norm.bias', 'mlp_AR.linear_1.weight', 'mlp_AR.linear_1.bias', 'mlp_AR.linear_2.weight', 'mlp_AR.linear_2.bias']
    if isinstance(allow_random_init_params, str): allow_random_init_params = allow_random_init_params.split(',')
    meta_sharded_sd = model.state_dict()
    sharded_sd = {}

    extra_meta_sharded_sd = set(meta_sharded_sd.keys()) - set((full_sd.keys()))
    extra_full_ds = set(full_sd.keys()) - set((meta_sharded_sd.keys()))
    extra_meta_sharded_sd = {
        k:(v.shape, v.device, v.dtype) for k, v in meta_sharded_sd.items() if k in extra_meta_sharded_sd
    }
    extra_full_ds = {
        k:(v.shape, v.device, v.dtype) for k, v in full_sd.items() if k in extra_full_ds
    }
    print(f"full_sd=\n{format_dict_or_list({k:(v.shape, v.device, v.dtype) for k, v in full_sd.items()})}")
    print(f"meta_sharded_sd=\n{format_dict_or_list({k:(v.shape, v.device, v.dtype) for k, v in meta_sharded_sd.items()})}")

    device0 = full_sd[list(full_sd)[0]]
    for k in extra_meta_sharded_sd:
        if allow_random_init_params is not None and k in allow_random_init_params:
            # full_sd[k] = meta_sharded_sd[k].clone()
            full_sd[k] = torch.rand(extra_meta_sharded_sd[k][0]) * 0.1 # ) .to(device0)
            if full_sd[k].ndim >= 2:
                nn.init.kaiming_normal_(full_sd[k], a=0, mode='fan_in', nonlinearity='relu')
            else:
                nn.init.zeros_(full_sd[k])  # 最常见
            full_sd[k] = full_sd[k].to(device0)
            # full_sd[k] = meta_sharded_sd[k].clone().to(device0)
            print(f"random init k={k}, {extra_meta_sharded_sd[k]}\n, meta_sharded_sd={meta_sharded_sd[k]} \nfull={full_sd[k]}")

    assert len(meta_sharded_sd) == len(full_sd), \
        f"Sharded State Dict doesn't equal to Full State Dict, {len(meta_sharded_sd) } v.s {len(full_sd)}" + "\n" + \
        f"extra_meta_sharded_sd={format_dict_or_list(extra_meta_sharded_sd)}, extra_full_ds={format_dict_or_list(extra_full_ds)}"
    assert sorted(list(meta_sharded_sd.keys())) == sorted(list(full_sd.keys())), \
        "Keys of Sharded State Dict doesn't equal to Full State Dict"


    for param_name, sharded_meta_param in meta_sharded_sd.items():
        full_tensor = full_sd[param_name].detach().cuda().type(sharded_meta_param.dtype)
        sharded_sd[param_name] = nn.Parameter(full_tensor)
    model.load_state_dict(sharded_sd, assign=True)


logits_all = []
if 1:
    # state_dict = load_hf_checkpoint(args.model_dir)
    try:
        with set_default_dtype(torch.bfloat16):
            model = Qwen3SiglipForConditionalGeneration_navit.from_pretrained(
                # "/llm_reco_ssd/zhouyang12/models/Qwen3-1.7B-siglip2",
                "/llm_reco/liuyang76/train_out/0.0.0/qwen3_2B_stage1/step1000/global_step1000/converted/",
                torch_dtype=torch.bfloat16,
                _attn_implementation = 'flash_attention_2',
                device_map="cuda:0",
                ignore_mismatched_sizes=True
            )

            # load_from_full_model_state_dict(model, load_hf_checkpoint("/llm_reco_ssd/zhouyang12/models/Qwen3-1.7B-siglip2"))
            # for k in inputst: inputst[k] = inputst[k].cuda()
            model = model.cuda()

            # logits = model(**inputst).logits
            # print("text_output", logits, logits.mean(), logits.max(), logits.min(), logits.shape)
            shapes = [(100,200),(100,100),(50, 1000), (400, 600), (465, 345), (155, 581), (201, 356), (34,532), (135,1799)]
            for a,b in shapes: 
                inputs = make_inputs(a,b)
                for k in inputs: inputs[k] = inputs[k].cuda()
                logits = model(**inputs).logits
                logits_all.append(logits)


    except Exception as e:
        import traceback
        traceback.print_exc()
        print(e)
        pass

for i, logits in enumerate(logits_all):
    print(f"mm_output-{i}", logits.flatten()[:4], logits.flatten().std(), logits.mean(), logits.max(), logits.min(), logits.shape)
    print(f"logits={logits[:30,:30,:30]}")