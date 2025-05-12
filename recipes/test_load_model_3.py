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

MODEL_DIR="/llm_reco_ssd/zhouyang12/models/Qwen3-8B-Base-siglip"
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
with set_default_dtype(torch.bfloat16):
    model = Qwen2_5_VLForConditionalGeneration_siglip.from_pretrained(
        MODEL_DIR,
        use_cache=False
    )




def debug_model_inference(model):
    # processor = Qwen2VLProcessor.from_pretrained(MODEL_DIR)
    MODEL_DIR2="/llm_reco_ssd/zhouyang12/models/Qwen3-8B-Base-siglip"
    processor = Qwen2_5_VLProcessor_siglip.from_pretrained(MODEL_DIR2)
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
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to(torch.cuda.current_device())
    model = model.to(torch.cuda.current_device())


    print_input_info({
        "inputs": inputs,
    })
    print_rank_0("=" * 100)

    output = model(**inputs); 
    logits = output.logits
    # Convert BFloat16 tensor to float32 before numpy conversion
    logits_np = logits.detach().cpu().float().numpy().tolist()
    json.dump(logits_np, open("logits1.json", "w"))
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print_rank_0(output_text)
    #print_rank_0(output)
    
    exit()
    generation_config = GenerationConfig(
    max_new_tokens=128,
    do_sample=True,
    top_p=0.95,
    temperature=0.7,
    )




debug_model_inference(model)