# /llm_reco_ssd/zhouyang12/models/Keye-8B-demo_hf_vit_rope_slowfast_0608/
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
from torch.distributed.checkpoint.default_planner import (
    _EmptyStateDictLoadPlanner
)




SHARD_FNAME = "model-{cpt_idx}-of-{num_shards}"

def dcp_to_torch_save(sd,
                      output_dir,
                      model_only: bool=True,
                      use_safetensor: bool=True,
                      max_gb_per_shard: int = 4,
                      model_type:str="Intern"):

  for k in tqdm.tqdm(sd):
    sd[k] = sd[k].to(torch.bfloat16)
  split_state_dicts: Dict[int, Dict[str, torch.Tensor]] = {}
  for key, value in tqdm.tqdm(sd.items()):
    split_state_dicts[key] = value
  
  split_state_dicts: Dict[int, Dict[str, torch.Tensor]] = {}
  cpt_idx = 0
  total_size = 0
  current_size = 0
  for key, weight in  tqdm.tqdm(sd.items()):
    if cpt_idx not in split_state_dicts:
      split_state_dicts[cpt_idx] = {}
    split_state_dicts[cpt_idx].update({key: weight})
    current_size += weight.numel() * weight.element_size()
    total_size += current_size
    if current_size >= max_gb_per_shard * 1024 * 1024 * 1024:
      cpt_idx += 1
      current_size = 0

  # write the partitioned state dicts to the right checkpoint file
  # e.g. model-00001-of-00004.safetensors, model-00002-of-00004.safetensors, etc
  num_shards = len(split_state_dicts)
  weight_map = {}
  for cpt_idx, model_state_dict in tqdm.tqdm(split_state_dicts.items()):
    # TODO: We should probably use the original shard name and just add a prefix
    # however, having the SHARD_FNAME standardizes our checkpoints
    shard_name = SHARD_FNAME.format(
      cpt_idx=f"{cpt_idx}".zfill(5), num_shards=f"{num_shards}".zfill(5)
    )
    output_path = Path(output_dir) / shard_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not use_safetensor:
      output_path = output_path.with_suffix(".bin")
      torch.save(model_state_dict, output_path)
    else:
      output_path = output_path.with_suffix(".safetensors")
      save_file(model_state_dict, output_path, metadata={"format": "pt"})
    for key, weight in model_state_dict.items():
      weight_map[key] = str(output_path.parts[-1])

    print(
      "Model checkpoint of size "
      f"{os.path.getsize(output_path) / 1024**3:.2f} GiB "
      f"saved to {output_path}"
    )
    
  if use_safetensor:
    weight_map_path = Path(output_dir) / "model.safetensors.index.json"
  else:
    weight_map_path = Path(output_dir) / "model.bin.index.json"
  with open(weight_map_path, "w") as f:
    f.write(json.dumps({
      "metadata": {
        "total_size": total_size
      },
      "weight_map": weight_map,
    }, indent=2))



