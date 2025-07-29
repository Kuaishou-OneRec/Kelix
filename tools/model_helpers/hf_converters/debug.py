import argparse
import re
import os
import glob
import tqdm
import torch
from typing import Union, Dict

import os
import json
import argparse
import torch
from pathlib import Path
from safetensors.torch import save_file
import transformers
from safetensors import safe_open
from safetensors.torch import save_file
# Qwen2VLForConditionalGeneration
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict
from torch.distributed.checkpoint.metadata import Metadata, STATE_DICT_TYPE
from torch.distributed.checkpoint.default_planner import    _EmptyStateDictLoadPlanner


def load_safe_tensors(path):
  d = {}
  for p in tqdm.tqdm(glob.glob(os.path.join(path, "*.safetensors"))):
    with safe_open(p, framework="pt", device="cpu") as f:
      for key in f.keys():
          d[key] = f.get_tensor(key)
  return d

# d = load_safe_tensors("/llm_reco_ssd/zhouyang12/models/Keye-2B-demo_fix_improc_slowfast/")

d = load_safe_tensors("/mmu_mllm_hdd_2/lingzhixin/models/Keye-32B-scratch_0606")


for k,v in d.items():
    if v.shape == torch.Size([1152, 3, 14, 14]):
        print(k, v.shape)

# /llm_reco_ssd/zangdunju/output2/RecoVLM/SigLIP/siglip_navit/global_step1000/model_float32.pth
d2 = torch.load("/llm_reco_ssd/zangdunju/output2/RecoVLM/SigLIP/siglip_navit/global_step1000/model_float32.pth")
# model.layers.17.self_attn.k_proj.bias 这个有bias



'''
['mlp_AR.linear_1.bias', 'mlp_AR.linear_1.weight', 'mlp_AR.linear_2.bias', 'mlp_AR.linear_2.weight', 'mlp_AR.pre_norm.bias', 'mlp_AR.pre_norm.weight', 'visual.vision_model.encoder.layers.0.self_attn.k_proj.bias', 'visual.vision_model.encoder.layers.0.self_attn.out_proj.bias', 'visual.vision_model.encoder.layers.0.self_attn.q_proj.bias', 'visual.vision_model.encoder.layers.0.self_attn.v_proj.bias', 'visual.vision_model.encoder.layers.1.self_attn.k_proj.bias', 'visual.vision_model.encoder.layers.1.self_attn.out_proj.bias', 'visual.vision_model.encoder.layers.1.self_attn.q_proj.bias', 'visual.vision_model.encoder.layers.1.self_attn.v_proj.bias', 'visual.vision_model.encoder.layers.10.self_attn.k_proj.bias', 'visual.vision_model.encoder.layers.10.self_attn.out_proj.bias', 'visual.vision_model.encoder.layers.10.self_attn.q_proj.bias', 'visual.vision_model.encoder.layers.10.self_attn.v_proj.bias', 'visual.vision_model.encoder.layers.11.self_attn.k_proj.bias', 'visual.vision_model.encoder.layers.11.self_attn.out_proj.bias', 'visual.vision_model.encoder.layers.11.self_attn.q_proj.bias', 'visual.vision_model.encoder.layers.11.self_attn.v_proj.bias', 'visual.vision_model.encoder.layers.12.self_attn.k_proj.bias', 'visual.vision_model.encoder.layers.12.self_attn.out_proj.bias', 'visual.vision_model.encoder.layers.12.self_attn.q_proj.bias', 'visual.vision_model.encoder.layers.12.self_attn.v_proj.bias', 'visual.vision_model.encoder.layers.13.self_attn.k_proj.bias', 'visual.vision_model.encoder.layers.13.self_attn.out_proj.bias', 'visual.vision_model.encoder.layers.13.self_attn.q_proj.bias', 'visual.vision_model.encoder.layers.13.self_attn.v_proj.bias', 'visual.vision_model.encoder.layers.14.self_attn.k_proj.bias', 'visual.vision_model.encoder.layers.14.self_attn.out_proj.bias', 'visual.vision_model.encoder.layers.14.self_attn.q_proj.bias', 'visual.vision_model.encoder.layers.14.self_attn.v_proj.bias', 'visual.vision_model.encoder.layers.15.self_attn.k_proj.bias', 'visual.vision_model.encoder.layers.15.self_attn.out_proj.bias', 'visual.vision_model.encoder.layers.15.self_attn.q_proj.bias', 'visual.vision_model.encoder.layers.15.self_attn.v_proj.bias', 'visual.vision_model.encoder.layers.16.self_attn.k_proj.bias', 'visual.vision_model.encoder.layers.16.self_attn.out_proj.bias', 'visual.vision_model.encoder.layers.16.self_attn.q_proj.bias', 'visual.vision_model.encoder.layers.16.self_attn.v_proj.bias', 'visual.vision_model.encoder.layers.17.self_attn.k_proj.bias', 'visual.vision_model.encoder.layers.17.self_attn.out_proj.bias', 'visual.vision_model.encoder.layers.17.self_attn.q_proj.bias', 'visual.vision_model.encoder.layers.17.self_attn.v_proj.bias', 'visual.vision_model.encoder.layers.18.self_attn.k_proj.bias', 'visual.vision_model.encoder.layers.18.self_attn.out_proj.bias', 'visual.vision_model.encoder.layers.18.self_attn.q_proj.bias', 'visual.vision_model.encoder.layers.18.self_attn.v_proj.bias', 'visual.vision_model.encoder.layers.19.self_attn.k_proj.bias', 'visual.vision_model.encoder.layers.19.self_attn.out_proj.bias', 'visual.vision_model.encoder.layers.19.self_attn.q_proj.bias', 'visual.vision_model.encoder.layers.19.self_attn.v_proj.bias', 'visual.vision_model.encoder.layers.2.self_attn.k_proj.bias', 'visual.vision_model.encoder.layers.2.self_attn.out_proj.bias', 'visual.vision_model.encoder.layers.2.self_attn.q_proj.bias', 'visual.vision_model.encoder.layers.2.self_attn.v_proj.bias', 'visual.vision_model.encoder.layers.20.self_attn.k_proj.bias', 'visual.vision_model.encoder.layers.20.self_attn.out_proj.bias', 'visual.vision_model.encoder.layers.20.self_attn.q_proj.bias', 'visual.vision_model.encoder.layers.20.self_attn.v_proj.bias', 'visual.vision_model.encoder.layers.21.self_attn.k_proj.bias', 'visual.vision_model.encoder.layers.21.self_attn.out_proj.bias', 'visual.vision_model.encoder.layers.21.self_attn.q_proj.bias', 'visual.vision_model.encoder.layers.21.self_attn.v_proj.bias', 'visual.vision_model.encoder.layers.22.self_attn.k_proj.bias', 'visual.vision_model.encoder.layers.22.self_attn.out_proj.bias', 'visual.vision_model.encoder.layers.22.self_attn.q_proj.bias', 'visual.vision_model.encoder.layers.22.self_attn.v_proj.bias', 'visual.vision_model.encoder.layers.23.self_attn.k_proj.bias', 'visual.vision_model.encoder.layers.23.self_attn.out_proj.bias', 'visual.vision_model.encoder.layers.23.self_attn.q_proj.bias', 'visual.vision_model.encoder.layers.23.self_attn.v_proj.bias', 'visual.vision_model.encoder.layers.24.self_attn.k_proj.bias', 'visual.vision_model.encoder.layers.24.self_attn.out_proj.bias', 'visual.vision_model.encoder.layers.24.self_attn.q_proj.bias', 'visual.vision_model.encoder.layers.24.self_attn.v_proj.bias', 'visual.vision_model.encoder.layers.25.self_attn.k_proj.bias', 'visual.vision_model.encoder.layers.25.self_attn.out_proj.bias', 'visual.vision_model.encoder.layers.25.self_attn.q_proj.bias', 'visual.vision_model.encoder.layers.25.self_attn.v_proj.bias', 'visual.vision_model.encoder.layers.26.self_attn.k_proj.bias', 'visual.vision_model.encoder.layers.26.self_attn.out_proj.bias', 'visual.vision_model.encoder.layers.26.self_attn.q_proj.bias', 'visual.vision_model.encoder.layers.26.self_attn.v_proj.bias', 'visual.vision_model.encoder.layers.3.self_attn.k_proj.bias', 'visual.vision_model.encoder.layers.3.self_attn.out_proj.bias', 'visual.vision_model.encoder.layers.3.self_attn.q_proj.bias', 'visual.vision_model.encoder.layers.3.self_attn.v_proj.bias', 'visual.vision_model.encoder.layers.4.self_attn.k_proj.bias', 'visual.vision_model.encoder.layers.4.self_attn.out_proj.bias', 'visual.vision_model.encoder.layers.4.self_attn.q_proj.bias', 'visual.vision_model.encoder.layers.4.self_attn.v_proj.bias', 'visual.vision_model.encoder.layers.5.self_attn.k_proj.bias', 'visual.vision_model.encoder.layers.5.self_attn.out_proj.bias', 'visual.vision_model.encoder.layers.5.self_attn.q_proj.bias', 'visual.vision_model.encoder.layers.5.self_attn.v_proj.bias', 'visual.vision_model.encoder.layers.6.self_attn.k_proj.bias', 'visual.vision_model.encoder.layers.6.self_attn.out_proj.bias', 'visual.vision_model.encoder.layers.6.self_attn.q_proj.bias', 'visual.vision_model.encoder.layers.6.self_attn.v_proj.bias', 'visual.vision_model.encoder.layers.7.self_attn.k_proj.bias', 'visual.vision_model.encoder.layers.7.self_attn.out_proj.bias', 'visual.vision_model.encoder.layers.7.self_attn.q_proj.bias', 'visual.vision_model.encoder.layers.7.self_attn.v_proj.bias', 'visual.vision_model.encoder.layers.8.self_attn.k_proj.bias', 'visual.vision_model.encoder.layers.8.self_attn.out_proj.bias', 'visual.vision_model.encoder.layers.8.self_attn.q_proj.bias', 'visual.vision_model.encoder.layers.8.self_attn.v_proj.bias', 'visual.vision_model.encoder.layers.9.self_attn.k_proj.bias', 'visual.vision_model.encoder.layers.9.self_attn.out_proj.bias', 'visual.vision_model.encoder.layers.9.self_attn.q_proj.bias', 'visual.vision_model.encoder.layers.9.self_attn.v_proj.bias', 'visual.vision_model.head.attention.out_proj.bias']


Traceback (most recent call last):
  File "/usr/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/usr/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "/llm_reco/lingzhixin/pub_models/models/versions/v0_8_1/Keye-32B/tests.py", line 152, in <module>
    model = AutoModel.from_pretrained(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/auto/auto_factory.py", line 559, in from_pretrained
    return model_class.from_pretrained(
  File "/usr/local/lib/python3.10/dist-packages/transformers/modeling_utils.py", line 262, in _wrapper
    return func(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/transformers/modeling_utils.py", line 4319, in from_pretrained
    ) = cls._load_pretrained_model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/modeling_utils.py", line 4897, in _load_pretrained_model
    new_error_msgs, offload_index, state_dict_index = _load_state_dict_into_meta_model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/modeling_utils.py", line 896, in _load_state_dict_into_meta_model
    set_module_tensor_to_device(model, param_name, param_device, **set_module_kwargs)
  File "/usr/local/lib/python3.10/dist-packages/accelerate/utils/modeling.py", line 310, in set_module_tensor_to_device
    raise ValueError(
ValueError: Trying to set a tensor of shape torch.Size([128]) in "weight" (which has shape torch.Size([80])), this look incorrect.



Traceback (most recent call last):
  File "/usr/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/usr/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "/llm_reco/lingzhixin/pub_models/models/versions/v0_8_1/Keye-32B/tests.py", line 152, in <module>
    model = AutoModel.from_pretrained(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/auto/auto_factory.py", line 559, in from_pretrained
    return model_class.from_pretrained(
  File "/usr/local/lib/python3.10/dist-packages/transformers/modeling_utils.py", line 262, in _wrapper
    return func(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/transformers/modeling_utils.py", line 4319, in from_pretrained
    ) = cls._load_pretrained_model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/modeling_utils.py", line 4897, in _load_pretrained_model
    new_error_msgs, offload_index, state_dict_index = _load_state_dict_into_meta_model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/modeling_utils.py", line 896, in _load_state_dict_into_meta_model
    set_module_tensor_to_device(model, param_name, param_device, **set_module_kwargs)
  File "/usr/local/lib/python3.10/dist-packages/accelerate/utils/modeling.py", line 310, in set_module_tensor_to_device
    raise ValueError(
ValueError: Trying to set a tensor of shape torch.Size([1152, 3, 14, 14]) in "weight" (which has shape torch.Size([1152, 3, 16, 16])), this look incorrect.
'''