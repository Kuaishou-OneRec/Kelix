from transformers import AutoModelForCausalLM, AutoTokenizer
from recovlm.models.qwen_3_vl.modeling_qwen3_vl import Qwen3_VLForConditionalGeneration_siglip
from recovlm.models.qwen_3_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor_siglip
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


# /llm_reco/lingzhixin/recovlm_qw0510/recovlm/recovlm/models/qwen3siglip/modeling_qwen3siglip.py
# from recovlm.models.qwen3siglip.modeling_qwen3siglip import Qwen3ForConditionalGeneration

from recovlm.utils.time_tracker import TimeTracker
from recipes.inspects import info_params_recursive

from transformers import AutoModelForCausalLM, AutoTokenizer


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
# torch.cuda.set_device(local_rank)
# torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)


MODEL_DIR="/llm_reco_ssd/zhouyang12/models/msy_Qwen3vl-8B-Base"
MODEL_DIR="/llm_reco_ssd/zhouyang12/models/Qwen3-8B-Base"
MODEL_DIR="/llm_reco_ssd/zhouyang12/models/Qwen3-8B-Base"

# MODEL_DIR="/llm_reco/lingzhixin/output2/RecoVLM-dev/Qwen2-VL-7B-run_sft_7B_fsdp_sp/0.0.5/_1000/global_step_1000_torch_ckpt/"

# Qwen2_5_VLForConditionalGeneration_siglip


model_name = "Qwen/Qwen3-8B"

messages = [
    {
        "role": "user",
        "content": "How are you"
        # [
        #     {"type": "text", "text": "How are you"},
        # ],
    }
]
processor = AutoTokenizer.from_pretrained("/llm_reco_ssd/zhouyang12/models/Qwen3-8B/")
# processor = Qwen2_5_VLProcessor_siglip.from_pretrained("/llm_reco_ssd/zhouyang12/models/Qwen3-8B-Base-siglip")

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
# image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    # images=image_inputs,
    # videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

print(inputs)
'''
{'input_ids': tensor([[151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13,
         151645,    198, 151644,    872,    198,   4340,    525,    498, 151645,
            198, 151644,  77091,    198]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}
'''
if 1:
    try:
        # from recovlm.qwen3.modeling_qwen3 import *
        with set_default_dtype(torch.float32):
            model = AutoModelForCausalLM.from_pretrained(
                "/llm_reco_ssd/zhouyang12/models/Qwen3-8B/",
                # "Qwen/Qwen3-1.7B",
                torch_dtype="auto",
                _attn_implementation = 'flash_attention_2',
                device_map="auto"
            )
            # model = model.float()
            logits = model(**inputs).logits
            print(222, logits, logits.shape)
            #
        with open("Qwen3ForConditionalGeneration_Qwen3-8B.txt", 'w') as f:
            f.write(info_params_recursive(model.model, max_depth=10))
            print(f"load is done")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(e)
        pass

    exit()
    print("=" *200)

    try:
        # from recovlm.qwen3.modeling_qwen3 import *
        with set_default_dtype(torch.float32):
            model = AutoModelForCausalLM.from_pretrained(
                "/llm_reco_ssd/zhouyang12/models/Qwen3-1.7B/",
                torch_dtype="auto",
                _attn_implementation = 'flash_attention_2',
                device_map="auto"
            )
            # model = model.float()
            logits = model(**inputs).logits
            print(222, logits, logits.shape)
            #
        with open("Qwen3ForConditionalGeneration_Qwen3-1.7B.txt", 'w') as f:
            f.write(info_params_recursive(model.model, max_depth=10))
            print(f"load is done")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(e)
        pass



'''
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

# 下载模型到本地
snapshot_download(
    repo_id="Qwen/Qwen3-1.7B",
    local_dir="/code/models/Qwen3-1.7B",
    revision="main",  # 使用最新版本
    token=None  # 如果需要认证，提供HF token
)



from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-1.7B"
# model_name = "/llm_reco_ssd/zhouyang12/models/Qwen3-1.7B/"
# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    cache_dir="/code/qwen3_1.7b"
)

# prepare the model input
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)

'''