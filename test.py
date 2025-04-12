import torch
from transformers import AutoTokenizer, AutoModel
from recovlm.models.intern_vl_3 import InternVLChatModel
from recovlm.training.common import set_default_dtype

import itertools

path = "/llm_reco/penghao03/intern-vl/InternVL3-2B"
with set_default_dtype(torch.bfloat16), torch.device("meta"):
    model = InternVLChatModel.from_pretrained(
            path,
            use_flash_attn=True)

for tensor in itertools.chain(model.parameters(), model.buffers()):
    assert tensor.device == torch.device("meta")
print(model.language_model.model.rotary_emb.inv_freq)
layer_num = 24
drop = 0.1
dpr = [x.item() for x in torch.linspace(0, drop, layer_num)]
print(dpr)
x_dpr = [drop * i / max(1, layer_num - 1) for i in range(layer_num)]
print(x_dpr)
# print(model)
# for n,p in model.named_parameters():
#     print(n,p.shape)

# model_path = '/llm_reco_ssd/zhouyang12/models/Qwen2-VL-7B-Instruct'
# qwen_model = AutoModel.from_pretrained(
#         model_path, _attn_implementation="flash_attention_2",
#         use_cache=False
# )
# print(qwen_model)
# for n,p in qwen_model.named_parameters():
#     print(n,p.shape)


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
# from recovlm.models.intern_vl_3 import InternVLChatModel
