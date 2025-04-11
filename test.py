import torch
from transformers import AutoTokenizer, AutoModel
path = "/llm_reco/penghao03/intern-vl/InternVL3-2B"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    use_flash_attn=True,
    trust_remote_code=True).eval().cuda()

import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np

from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# from transformers import AutoTokenizer, AutoProcessor
from recovlm.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
from recovlm.models.qwen2_vl import Qwen2VLForConditionalGeneration

#inten-vl
from recovlm.models.intern_vl_3 import InternVLChatModel
