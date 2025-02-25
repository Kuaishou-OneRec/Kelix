
from typing import Dict, Any, Union, Optional
import collections

import re
import os
import gc
import glob
import torch
from safetensors import safe_open
import torch.distributed as dist
import deepspeed

from recovlm.utils import print_rank_0

def load_safetensors(path):
  tensors = {}
  with safe_open(path, framework="pt", device="cpu") as f:
    for k in f.keys():
      tensors[k] = f.get_tensor(k)
  return tensors


def load_zero3_state_dict(model, model_dir):

  missing_keys: List[str] = []
  unexpected_keys: List[str] = []
  error_msgs: List[str] = []

  if dist.get_rank() == 0:
    state_dict = collections.OrderedDict()
    patterns = glob.glob(os.path.join(model_dir, "model-*.safetensors"))
    for model_path in patterns:
      # state_dict.update(torch.load(model_path, map_location="cpu"))
      state_dict.update(load_safetensors(model_path))

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    if metadata is not None:
      # mypy isn't aware that "_metadata" exists in state_dict
      state_dict._metadata = metadata  # type: ignore[attr-defined]
  else:
    state_dict = None

  dist.barrier()

  def load(module, local_state_dict, prefix=""):
    # because zero3 puts placeholders in model params, this context
    # manager gathers (unpartitions) the params of the current layer, then loads from
    # the state dict and then re-partitions them again
    with deepspeed.zero.GatheredParameters(list(module.parameters(recurse=False)), modifier_rank=0):
      if dist.get_rank() == 0:
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        print_rank_0(f"Load: {prefix}")
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True,
            missing_keys, unexpected_keys, error_msgs
        )

    for name, child in module._modules.items():
      if child is not None:
        child_prefix = prefix + name + '.'
        if state_dict:
          child_state_dict = {
              k: v for k, v in local_state_dict.items() if
              k.startswith(child_prefix)}
        else:
          child_state_dict = None
        load(child, child_state_dict, child_prefix)

  load(model, state_dict, prefix="")

def load_dist_attn_state_dict(src, dst):
  # src: state_dict
  # dst: module
  new_state_dict = collections.OrderedDict()
  for k, v in src.items():
    if re.match(r"model.layers.(\d+).self_attn.*", k):
      new_k = re.sub(r'self_attn', 'self_attn.local_attn', k)
      print_rank_0(f"Replace key from {k} to {new_k}")
      k = new_k
    new_state_dict[k] = v
  dst.load_state_dict(new_state_dict, strict=True)

def safe_torch_load(
    checkpoint_path: Union[Path, str], weights_only: bool = True, mmap: bool = True) -> Dict[str, Any]:
    """
    Utility to load a checkpoint file onto CPU in a safe manner. Provides separate handling for
    safetensors files.

    Args:
        checkpoint_path (Union[Path, str]): Path to the checkpoint file.
        weights_only (bool): Whether to load only tensors, primitive types, and dictionaries
            (passthrough to torch.load). Default: True
        mmap (bool): Whether to mmap from disk into CPU memory. Default: True

    Returns:
        Dict[str, Any]: State dict from the checkpoint file.

    Raises:
        ValueError: If the checkpoint file is not found or cannot be loaded.
    """
    try:
        # convert the path into a string since pathlib Path and mmap don't work
        # well together
        is_safetensors_file = (
            True if str(checkpoint_path).endswith(".safetensors") else False
        )
        if is_safetensors_file:
            result = {}
            from safetensors import safe_open
            with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
                for k in f.keys():
                    result[k] = f.get_tensor(k)
            state_dict = result
        else:
            state_dict = torch.load(
                str(checkpoint_path),
                map_location="cpu",
                mmap=mmap,
                weights_only=weights_only,
            )
    except Exception as e:
        raise ValueError(f"Unable to load checkpoint from {checkpoint_path}. ") from e
    return state_dict

def load_hf_checkpoint(model_dir):
  # merged state_dict contains keys and weights from all the checkpoint files
  merged_state_dict: Dict[str, torch.Tensor] = {}

  # converted_state_dict is the final state_dict passed to the recipe after the
  # keys are converted into the torchtune format. This optionally also contains
  # the recipe state and adapter weights
  ckpt_paths = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
  if not ckpt_paths:
    ckpt_paths = sorted(glob.glob(os.path.join(model_dir, "*.bin")))
  # _checkpoint_paths are already sorted so simply enumerate to generate the right id
  for cpt_idx, cpt_path in enumerate(ckpt_paths):
    print_rank_0(f"Load checkpoints: {cpt_idx}/{len(ckpt_paths)}")
    state_dict = safe_torch_load(cpt_path)
    for key, value in state_dict.items():
        # Ensure that the state dict is a flat dict of keys and tensors. Breaking this assumption
        # will break recipe code
        if not isinstance(value, torch.Tensor):
            raise ValueError(
                f"Expected all values in the state dict to be torch.Tensor. "
                f"Found {key}={type(value)} instead."
            )
    merged_state_dict.update(state_dict)

    # delete the state_dict to free up memory; TODO check if this del is needed
    del state_dict
    gc.collect()
  return merged_state_dict
