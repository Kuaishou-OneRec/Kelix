from recovlm.models.qwen_3_vl_2.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration_siglip
from recovlm.models.qwen_3_vl_2.processing_qwen2_5_vl import Qwen2_5_VLProcessor_siglip
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
import multiprocessing as mp
from functools import partial

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

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np

from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from transformers import AutoTokenizer


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
from recovlm.models.qwen_2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLDecoderLayer, Qwen2_5_VLVisionBlock
from recovlm.utils.time_tracker import TimeTracker



# # initialize distributed environment
# rank = int(os.environ["RANK"])
# world_size = int(os.environ.get("WORLD_SIZE", "1"))
# local_rank = int(os.environ.get("LOCAL_RANK", "0"))
# device = torch.device(f"cuda:{local_rank}")
# torch.cuda.set_device(device)
# torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)

rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", 0))
world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", 0))
local_rank = rank % world_size

# print_rank_0(rank, world_size, local_rank)
# torch init
torch.cuda.set_device(local_rank)
torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)

initialize_model_parallel(1)

MODEL_DIR="/llm_reco_ssd/zhouyang12/models/msy_Qwen3vl-8B-Base"
# MODEL_DIR="/llm_reco/lingzhixin/output2/RecoVLM-dev/Qwen2-VL-7B-run_sft_7B_fsdp_sp/0.0.5/_1000/global_step_1000_torch_ckpt/"


state_dict = None
if dist.get_rank() == 0:
  with set_default_dtype(torch.bfloat16):
    state_dict = load_hf_checkpoint(MODEL_DIR)
    # converter = Qwen2VLCheckpointConverter(MODEL_DIR)
    converter = Qwen2_5_VL_siglipCheckpointConverter(MODEL_DIR)
    state_dict = converter(state_dict)

dist.barrier()

# Load model in meta mode to avoid OOM during initialization
with set_default_dtype(torch.float32), torch.device("meta"):
    model = Qwen2_5_VLForConditionalGeneration_siglip.from_pretrained(
        MODEL_DIR,
        _attn_implementation="flash_attention_2",
        use_cache=False
    )
    # state_dict = torch.load("/llm_reco/maosiyang/model/qwen_moonvit/qwen3_vl_siglip_state_dict_2.pth", weights_only=True)
    # model.load_state_dict(state_dict)

device_mesh = init_device_mesh("cuda", mesh_shape=(dist.get_world_size(),))

for tensor in itertools.chain(model.parameters(), model.buffers()):
    assert tensor.device == torch.device("meta")

model = model.float()
shard_model(
    model=model,
    shard_conditions=[partial(get_shard_conditions, model_class='Qwen2_5_VLForConditionalGeneration_siglip')],
    cpu_offload=False,
    reshard_after_forward=False,
    dp_mesh=device_mesh,
    fp32_weight=True,
    model_class='Qwen2_5_VLForConditionalGeneration_siglip',
    fp32_reduce=True
)
dist.barrier()
load_from_full_model_state_dict(model=model, full_sd=state_dict)

with torch.device(torch.cuda.current_device()):
    for m in model.modules():
        # RoPE is not covered in state dict
        if hasattr(m, "rope_init"):
            m.rope_init()


print_rank_0(model)

for name, param in model.named_parameters():
    print_rank_0(name, param.device, param.shape, type(param))


from PIL import Image, ImageDraw


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


def debug_model_inference(model):
    # processor = Qwen2VLProcessor.from_pretrained(MODEL_DIR)
    processor = Qwen2_5_VLProcessor_siglip.from_pretrained(MODEL_DIR)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Give me a short introduction to large language model."},
            ],
        }
    ]
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # image_inputs, video_inputs = process_vision_info(messages)
    print_rank_0(text)
    inputs = processor(
        text=[text],
        # images=image_inputs,
        # videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to(torch.cuda.current_device())


    print_input_info({
        "inputs": inputs,
    })
    print_rank_0("=" * 100)
    # output = model(
    #     inputs["input_ids"],
    #     # pixel_values=inputs["pixel_values"],
    #     # image_grid_thw=inputs["image_grid_thw"]
    # )

    # print_rank_0(output)

    # add generation config

    output = model(**inputs); 
    logits = output.logits
    # Convert BFloat16 tensor to float32 before numpy conversion
    logits_np = logits.detach().cpu().float().numpy().tolist()
    json.dump(logits_np, open("logits1.json", "w"))

    #print_rank_0(output)
    
    exit()
    generation_config = GenerationConfig(
    max_new_tokens=128,
    do_sample=True,
    top_p=0.95,
    temperature=0.7,
    )




debug_model_inference(model)