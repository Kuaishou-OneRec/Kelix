import torch
import os
#print(os.environ.get("CUDA_VISIBLE_DEVICES", None)); exit()

# 配置打印选项
torch.set_printoptions(
    precision=6,        # 保留6位小数
    sci_mode=False,     # 禁用科学记数法
    linewidth=200,     # 设置每行最大字符数，防止矩阵被截断
    threshold=12   # 设置显示元素的最大数量，避免只显示省略号
)



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

from transformers import AutoTokenizer, AutoModel, AutoProcessor
from recovlm.models.keye_vitrope_slowfast.keye_vl_utils import process_vision_info


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
# from recovlm.utils.qwen_vl_utils import process_vision_info

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
# from recovlm.models.qwen_2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLDecoderLayer, Qwen2_5_VLVisionBlock

from recovlm.models.keye_vitrope_slowfast.modeling_keye import KeyeForConditionalGeneration as KeyeForConditionalGeneration_vitrope_slowfast

from recovlm.utils.time_tracker import TimeTracker
from recipes.inspects import info_params_recursive



rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", 0))
world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", 0))
local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0))
local_world_size = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_SIZE", 0))




MODEL_DIR="/llm_reco_ssd/zhouyang12/models/msy_Qwen3vl-8B-Base"
MODEL_DIR="/llm_reco_ssd/zhouyang12/models/Qwen3-8B-Base-siglip"
MODEL_DIR="/mmu_mllm_hdd_2/zhouyang12/models/Keye-8B-demo_hf_vit_rope_slowfast_0714_sp1"
MODEL_DIR="/mmu_mllm_hdd_2/zhouyang12/output1/Keye/0.9.3/Stage2/8b/slowfast-0721-0717-v2/step53000/global_step57000/converted"
# MODEL_DIR="/llm_reco/lingzhixin/output2/RecoVLM-dev/Qwen2-VL-7B-run_sft_7B_fsdp_sp/0.0.5/_1000/global_step_1000_torch_ckpt/"

# Qwen2_5_VLForConditionalGeneration_siglip

# print_rank_0(model)

# for name, param in model.named_parameters():
#     print_rank_0(name, param.device, param.shape, type(param))


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
    #     processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
    s = 15

    messages = [
        # {
        #     "role": "user",
        #     "content": [
        #         {"type": "text", "text": "How are you Give me a short introduction to large language model."},
        #     ],
        # }
        # 44 15 15 4 1
        {
            "role": "user",
            "content": [
                {"type": "image", "image": generate_circle_image((s,s))},
                {"type": "image", "image": generate_circle_image((s,s))},
                {"type": "image", "image": generate_circle_image((s,s))},
                {"type": "image", "image": generate_circle_image((s,s))},
                # {"type": "text", "text": "Once Upon a Time There Was a Spirit Sword Mountain is a hilarious fantasy 修仙 (xiuxian, immortal cultivation) story that subverts traditional. I am Optimus Prime !!!, How dare you?"},
                {"type": "text", "text": "a,"},
            ],
        }
    ]
    is_split1 = os.environ.get("is_split1", False)
    print("is_split1is_split1", is_split1)
    if is_split1:
        messages = [
          # 44 15 15 4 1
          {
              "role": "user",
              "content": [
                  {"type": "image", "image": generate_circle_image((s,s))},
              ],
          }
      ]
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    ).split("<|im_start|>user")[1].split("<|im_end|>")[0]
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        # videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    print(343333434214254)
    #print(inputs["input_ids"].shape, inputs["pixel_values"].shape, inputs["fast_pixel_values"].shape)
    if dist.get_rank() == 0:
      print("=" * 10)
      print(text)
      print(inputs["input_ids"][0].tolist())
      print("=" * 10)

    # print(inputs["input_ids"].shape); exit()
    inputs = inputs.to(torch.cuda.current_device())
    inputs["cu_seqlens"] = torch.Tensor([0,inputs["input_ids"].shape[1]]).to(model.device).long()
    # print_input_info({
    #     "inputs": inputs,
    # })
    print_rank_0("=" * 100)
    # output = model(
    #     inputs["input_ids"],
    #     # pixel_values=inputs["pixel_values"],
    #     # image_grid_thw=inputs["image_grid_thw"]
    # )

    # print_rank_0(output)

    # add generation config
    
    print(f"beforerrrrrr", model.dtype, id(model))
    output = model(**inputs); 
    logits = output.logits
    # Convert BFloat16 tensor to float32 before numpy conversion
    # logits_np = logits.detach().cpu().float().numpy().tolist()
    # json.dump(logits_np, open("logits1.json", "w"))
    del inputs["cu_seqlens"]
    # generated_ids = model.generate(**inputs, max_new_tokens=128, top_k=1)
    # generated_ids_trimmed = [
    #     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    # ]
    # output_text = processor.batch_decode(
    #     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    # )
    if dist.get_rank() == 0:
        print_input_info({
            "inputs": inputs,
            "logits": logits[:, :6],
            # "output_text": output_text
        })
    
    print('=' * 30)
    if dist.get_rank() in [0, 7]:
        print(f"rank{dist.get_rank()}", logits.shape, "\n", logits[:6])
    # print_rank_0(output_text)


if __name__=='__main__':
  # 禁用 CuDNN 的非确定性算法（确保结果可复现）
  torch.backends.cudnn.deterministic = True
  # 禁用 CuDNN 的自动调优功能（确保每次运行使用相同的算法）
  torch.backends.cudnn.benchmark = False

  torch.cuda.set_device(local_rank)
  torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)

  initialize_model_parallel(int(os.environ.get("SP_N", 8)))



  state_dict = None


  dist.barrier()

  # Load model in meta mode to avoid OOM during initialization
  with set_default_dtype(torch.float32), torch.device("meta"):
      model = KeyeForConditionalGeneration_vitrope_slowfast.from_pretrained(
          MODEL_DIR,
          _attn_implementation="flash_attention_2",
          use_cache=False,
          device_map="meta",
          torch_dtype="float32",
      )
      # state_dict = torch.load("/llm_reco/maosiyang/model/qwen_moonvit/qwen3_vl_siglip_state_dict.pth", weights_only=True)

  # with open("KeyeForConditionalGeneration_vitrope_slowfast_0702_1738.txt", 'w') as f:
  #     f.write(info_params_recursive(model, max_depth=10))
  device_mesh = init_device_mesh("cuda", mesh_shape=(dist.get_world_size(),))
  state_dict = load_hf_checkpoint(MODEL_DIR)

  for tensor in itertools.chain(model.parameters(), model.buffers()):
      assert tensor.device == torch.device("meta")

  model = model.float()
  shard_model(
      model=model,
      shard_conditions=[partial(get_shard_conditions, model_class='KeyeForConditionalGeneration_vitrope_slowfast')],
      cpu_offload=False,
      reshard_after_forward=False,
      dp_mesh=device_mesh,
      fp32_weight=True,
      model_class='KeyeForConditionalGeneration_vitrope_slowfast',
      fp32_reduce=True,
      param_dtype=torch.bfloat16
  )
  dist.barrier()
  load_from_full_model_state_dict(model=model, full_sd=state_dict, allow_random_init_params='mlp_AR.pre_norm.weight,mlp_AR.pre_norm.bias,mlp_AR.linear_1.weight,mlp_AR.linear_1.bias,mlp_AR.linear_2.weight,mlp_AR.linear_2.bias,visual_fast.vision_model.embeddings.packing_position_embedding.weight,fast_mlp_AR.pre_norm.weight,fast_mlp_AR.pre_norm.bias,fast_mlp_AR.linear_1.weight,fast_mlp_AR.linear_1.bias,fast_mlp_AR.linear_2.weight,fast_mlp_AR.linear_2.bias')
  with torch.device(torch.cuda.current_device()):
      for m in model.modules():
          # RoPE is not covered in state dict
          if hasattr(m, "rope_init"):
              m.rope_init()

  # model = model.float()
  for param in model.parameters():
      print(param.dtype)  # 应该输出torch.float32
  debug_model_inference(model)



'''
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
=================================================================================================================================
==================================================testing Nonewith_sys=True ...==================================================
=================================================================================================================================
{
  "role": "user",
  "content": "</think>

A large language model (LLM) is a type of artificial intelligence that is trained on vast amounts of text data to understand and generate human-like text. These models can perform a wide range of tasks, such as answering questions, writing stories, coding, and translating languages. LLMs are powered by deep learning techniques and are capable of processing and generating text at a scale and complexity that was previously unattainable. They are widely used in various applications, including chatbots, content creation, and data analysis.",
  "input_ids": torch.Size([1, 26]),
  "pixel_values": None,
  "pixel_values_videos": None,
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Give me a short introduction to large language model."
        }
      ]
    }
  ],
  "logits": tensor([[[ 4.1875,  3.3906,  2.8750,  ...,  0.5312,  0.5312,  0.5312],
         [ 4.9688,  5.8125,  5.9062,  ..., -1.0312, -1.0312, -1.0312],
         [ 0.3770,  5.7812, 11.6875,  ...,  3.4219,  3.4219,  3.4219],
         ...,
         [ 4.5000, -3.0469,  3.1406,  ...,  2.7188,  2.7188,  2.7188],
         [ 6.4688,  2.2969,  1.4375,  ...,  6.9062,  6.9062,  6.9062],
         [ 8.4375,  8.6875,  3.6406,  ...,  8.6875,  8.6875,  8.6875]]],
       device='cuda:0', grad_fn=<UnsafeViewBackward0>),
  "logits.shape": torch.Size([1, 26, 151936])
}
==================================================================================================================================
==================================================testing Nonewith_sys=False ...==================================================
==================================================================================================================================
{
  "role": "user",
  "content": ".</think>

A large language model (LLM) is an advanced type of artificial intelligence that is trained on vast amounts of text data to understand and generate human-like text. These models can perform a wide range of tasks, such as answering questions, writing stories, coding, and translating languages. LLMs are built using deep learning techniques and have the ability to process and generate text at a scale and complexity that was previously unattainable. They are widely used in various applications, including chatbots, virtual assistants, and content creation tools.",
  "input_ids": torch.Size([1, 26]),
  "pixel_values": None,
  "pixel_values_videos": None,
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Give me a short introduction to large language model."
        }
      ]
    }
  ],
  "logits": tensor([[[ 4.1875,  3.3906,  2.8750,  ...,  0.5312,  0.5312,  0.5312],
         [ 4.9688,  5.8125,  5.9062,  ..., -1.0312, -1.0312, -1.0312],
         [ 0.3770,  5.7812, 11.6875,  ...,  3.4219,  3.4219,  3.4219],
         ...,
         [ 4.5000, -3.0469,  3.1406,  ...,  2.7188,  2.7188,  2.7188],
         [ 6.4688,  2.2969,  1.4375,  ...,  6.9062,  6.9062,  6.9062],
         [ 8.4375,  8.6875,  3.6406,  ...,  8.6875,  8.6875,  8.6875]]],
       device='cuda:0', grad_fn=<UnsafeViewBackward0>),
  "logits.shape": torch.Size([1, 26, 151936])
}
'''
