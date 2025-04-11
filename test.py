import torch
from transformers import AutoTokenizer, AutoModel
from recovlm.models.intern_vl_3 import InternVLChatModel
from recovlm.training.common import set_default_dtype
path = "/llm_reco/penghao03/intern-vl/InternVL3-2B"

# model = InternVLChatModel.from_pretrained(
#     path,use_flash_attn=True,device_map = 'balanced')

#print(model._tp_plan)

#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('OpenGVLab/InternVL2_5-2B',cache_dir='/llm_reco/penghao03/intern-vl/InternVL2_5-2B')

# import torch.nn as nn
# import torch.distributed as dist
# import torch.nn.functional as F
# import numpy as np

# from pathlib import Path
# from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter

# # from transformers import AutoTokenizer, AutoProcessor
# from recovlm.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
# from recovlm.models.qwen2_vl import Qwen2VLForConditionalGeneration

# #inten-vl

