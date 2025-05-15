from recovlm.models.qwen3siglip.modeling_qwen3siglip import Qwen3SiglipForConditionalGeneration_navit
from recovlm.models.qwen3siglip.processing_qwen3siglip import Qwen3SiglipProcessor_siglip
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

MODEL_DIR="/llm_reco_ssd/zhouyang12/models/Qwen3-1.7B-siglip"
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
    model = Qwen3SiglipForConditionalGeneration_navit.from_pretrained(
        MODEL_DIR,
        _attn_implementation = 'flash_attention_2',
        use_cache=False
    )




def debug_model_inference(model):
    MODEL_DIR2="/llm_reco_ssd/zhouyang12/models/Qwen3-1.7B-siglip"
    processor = Qwen3SiglipProcessor_siglip.from_pretrained(MODEL_DIR2)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Give me a short introduction to large language model."},
            ],
        },
        {
            "role": "assistant",
            "content":[
                {
                  "type":"text",
                  "text":"A large language model (LLM) is an AI system lalallallalal alal al a a ll alla l al la l a"
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What are its main applications?"},
            ],
        },
        {
            "role": "assistant",
            "content":[
                {
                  "type":"text",
                  "text":"LLMs are used for tasks like text generation, translation, summarization, and question answering."
                }
            ]
        }
    ]
    
    # Get the full conversation text
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # Tokenize the full conversation
    inputs = processor(
        text=[text],
        padding=True,
        return_tensors="pt",
    )
    
    # Move to GPU
    inputs = inputs.to(torch.cuda.current_device())
    model = model.to(torch.cuda.current_device())

    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the logits and input ids
    logits = outputs.logits
    input_ids = inputs["input_ids"]
    
    # Find all assistant responses in the messages
    assistant_responses = []
    for i, msg in enumerate(messages):
        if msg["role"] == "assistant":
            assistant_responses.append({
                "text": msg["content"][0]["text"],
                "index": i
            })
    
    # Calculate PPL for each assistant response
    total_ppl = 0
    for assistant_response in assistant_responses:
        # Get assistant's response text
        assistant_text = assistant_response["text"]
        
        # Tokenize the assistant's response
        assistant_tokens = processor(
            text=[assistant_text],
            padding=True,
            return_tensors="pt",
        )["input_ids"][0]
        assistant_tokens = assistant_tokens.to(torch.cuda.current_device())
        
        # Find the start position of this assistant's response in the full sequence
        assistant_start_pos = None
        for i in range(len(input_ids[0]) - len(assistant_tokens)):
            if torch.all(input_ids[0][i:i+len(assistant_tokens)] == assistant_tokens):
                assistant_start_pos = i
                break
        
        if assistant_start_pos is None:
            print_rank_0(f"Could not find assistant's response in the sequence: {assistant_text}")
            continue
        
        # Calculate PPL for this assistant's response
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        # Calculate loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.view(shift_logits.size(0), -1)
        
        # Calculate PPL for this response
        assistant_loss = loss[0, assistant_start_pos-1:assistant_start_pos+len(assistant_tokens)-1]
        response_ppl = torch.exp(assistant_loss.mean())
        
        print_rank_0(f"\nAssistant response {assistant_response['index']}: {assistant_text}")
        print_rank_0(f"Perplexity: {response_ppl.item():.4f}")
        
        total_ppl += response_ppl.item()
    
    # Calculate average PPL across all responses
    avg_ppl = total_ppl / len(assistant_responses)
    print_rank_0(f"\nFull conversation: {text}")
    print_rank_0(f"Average Perplexity across all responses: {avg_ppl:.4f}")
    
    return avg_ppl

debug_model_inference(model)