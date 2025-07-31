from typing import Union, Dict

import os
import json
import argparse
import torch
from pathlib import Path
from safetensors.torch import save_file
import tqdm
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict
from torch.distributed.checkpoint.metadata import Metadata, STATE_DICT_TYPE
from torch.distributed.checkpoint.default_planner import (
    _EmptyStateDictLoadPlanner
)
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import re


SHARD_FNAME = "model-{cpt_idx}-of-{num_shards}"

def dcp_to_torch_save(dcp_checkpoint_dir: Union[str, os.PathLike],
                      output_dir: Union[str, os.PathLike],
                      model_only: bool=True,
                      use_safetensor: bool=True,
                      max_gb_per_shard: int = 4,
                      model_type:str="Intern"):
  """
    Given a directory containing a DCP checkpoint, this function will convert it into a
    Torch save file.

    Args:
        dcp_checkpoint_dir: Directory containing the DCP checkpoint.
        torch_save_path: Filename to store the converted Torch save file, e.g., 
            /path/to/model/pytorch_model.bin
        model_only: Save model weights only

    .. warning::
        To avoid OOM, it's recommended to only run this function on a single rank.
  """
  sd: STATE_DICT_TYPE = {}
  import os
  print(f"dcp_to_torch_save({os.getpid()}): _load_state_dict ...")
  _load_state_dict(
        sd,
        storage_reader=FileSystemReader(dcp_checkpoint_dir),
        planner=_EmptyStateDictLoadPlanner(),
        no_dist=True,
  )

  print(f"dcp_to_torch_save: _load_state_dict done.")
  # if listinstr(["qwen"], model_type.lower()):
  #   sd = Qwen2VLCheckpointConverter().tp_to_original(sd)
  if model_only:
    sd = sd["app"]["model"]


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

def get_argument_parser():
  parser = argparse.ArgumentParser()

  ############ Checkpoint args ############
  parser.add_argument("--checkpoint_dir", type=str, default=None,
                      help="The directory of the pretrained model.")

  parser.add_argument("--output_dir", type=str, default=None,
                      help="The directory of the pretrained model.")

  return parser

def main():
  arg_parser = get_argument_parser()
  args = arg_parser.parse_args()
  dcp_to_torch_save(
    dcp_checkpoint_dir=args.checkpoint_dir,
    output_dir=args.output_dir,
    model_only=True,
    use_safetensor=True,
    max_gb_per_shard=5
  )

if __name__ == "__main__":
  main()