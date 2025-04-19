from typing import Dict, Any, Union, Optional, List, Protocol
import collections
from recovlm.utils.ds_utils import print_input_info
import re
import os
import gc
import glob
import time
import torch
from pathlib import Path
from safetensors import safe_open
import torch.distributed as dist
import deepspeed
from concurrent.futures import Future

from recovlm.training.distributed import get_world_size_and_rank
from recovlm.utils.common import print_rank_0, print_rank_n

from torch.distributed.checkpoint import (
    async_save,
    FileSystemReader,
    FileSystemWriter,
    load,
    save,
)

from torch.distributed.checkpoint.metadata import Metadata, STATE_DICT_TYPE
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict, get_model_state_dict, set_model_state_dict


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
      
def gather_cpu_state_dict(
    model: "FSDPModule",  # noqa
    is_rank_zero: bool
) -> Dict[str, Any]:
    """
    Converting sharded state dict into a full state dict on CPU
    Returning non-empty result only on rank0 to avoid peaking CPU memory
    Currenltly we can used distributed state dict API to process model without NF4Tensor. Otherwise, we need to
    manually gather any NF4 tensors until all-gather is supported in the NF4Tensor subclass
    TODO: add support for NF4Tensor at distributed state dict API

    Args:
        model (FSDPModule): Model to generate fully qualified names for cpu_state_dict
        is_rank_zero (bool): flag to check if the process is on rank 0
        device (Optional[torch.device]): device to use for sharded tensors. Default: None

    Returns:
        Dict[str, Any]: State dict on CPU
    """
    # TODO: Disabling DSD as it has issues. Add back changes in #2138 once DSD issue is fixed.
    cpu_state_dict = {}
    sharded_sd = model.state_dict()
    for param_name, param in sharded_sd.items():
      if hasattr(param, "_local_tensor"):
        param = param.full_tensor()
      if is_rank_zero:
        cpu_state_dict[param_name] = param.cpu()
      torch.distributed.barrier()
    return cpu_state_dict

class CheckpointerInterface(Protocol):
  def load_checkpoint(self, **kwargs) -> Dict[str, Any]:
    ...

  def save_checkpoint(self, state_dict: Dict[str, Any], **kwargs) -> None:
    ...

class HFCheckpointer(CheckpointerInterface):
  pass

class DistributedCheckpointer(CheckpointerInterface):
  """
  Checkpointer which reads and writes checkpoints in the DistributedCheckpointing format.

  Args:
    checkpoint_dir (str): Directory containing the checkpoint files
    output_dir (str): Directory to save the checkpoint files
    process_group (Optional[dist.ProcessGroup]): Optional process group to use
        for distributed saving/loading. If None, the default process group will be used.
        For checkpointing, gloo CPU-based backend is needed.
  """

  def __init__(
      self,
      process_group: Optional[dist.ProcessGroup] = None) -> None:
    self._checkpoint_future = None
    self._checkpoint_dir_prefix = "global_step"
    _, self._rank = get_world_size_and_rank()
    self._process_group: Optional[dist.ProcessGroup] = process_group

  def get_latest_checkpoint(self, checkpoint_dir: str):
    checkpoint_dir_pattern = re.compile(f"{self._checkpoint_dir_prefix}(\\d+)")
    checkpoint_paths = [
        name
        for name in os.listdir(checkpoint_dir)
        if re.match(checkpoint_dir_pattern, name)
        and os.path.isfile(
            os.path.join(self._output_dir, name, self._metadata_file)
        )
    ]

    if checkpoint_paths:
      latest_checkpoint_dir = sorted(
        checkpoint_paths, key=lambda x: int(x.split("_")[-1])
      )[-1]
      return os.path.join(self._output_dir, latest_checkpoint_dir)
    return None

  def load_checkpoint(self,
                      state_dict: STATE_DICT_TYPE,
                      checkpoint_path: Optional[str] = None,
                      checkpoint_dir: Optional[str] = None,
                      tag: Union[str, int] = "latest") -> Dict[str, Any]:
    """
    Load a Distributed checkpoint saved at the <checkpoint_path>
    If no path is provided, latest intermediate checkpoint is loaded.
    """
    if not checkpoint_path:
      assert checkpoint_dir and tag, \
        "checkpoint_dir and tag should be provided if checkpoint_path is None"
      if tag == "latest":
        checkpoint_path = self.get_latest_checkpoint(checkpoint_dir)
      else:
        checkpoint_path = Path(checkpoint_dir) / str(tag)

    if not checkpoint_path:
      raise ValueError("No checkpoint path provided.")

    print_rank_0(f"Loading checkpoint from {checkpoint_path}")

    print_input_info(state_dict)
    dcp.load(
      state_dict=state_dict,
      storage_reader=FileSystemReader(checkpoint_path),
      process_group=self._process_group,
    )

    return state_dict

  def save_checkpoint(
      self,
      state_dict: STATE_DICT_TYPE,
      output_dir,
      tag: Optional[Union[str, int]] = None,
      save_async: bool = False) -> None:
    """
    Save a distributed checkpoint to storage.
    If ``save_async`` is True, the save happens asynchronously unblocking the GPUs sooner. This
    should only be used for the intermediate checkpoints. Final checkpoint has to be a synchronous
    one as the finetuning job can not terminate until the checkpoint gets persisted.

    Args:
      state_dict (Dict[str, Any]): Checkpoint state dict to be written out to file
      tag (int): Checkpoint tag. Used to create the checkpoint file name, generally step
      save_async (bool): If True, save the checkpoint asynchronously
    """
    checkpoint_path = output_dir
    if tag is not None:
      checkpoint_path = Path(output_dir) / f"{self._checkpoint_dir_prefix}{tag}"
    print_rank_0(f"Saving checkpoint to {checkpoint_path}")

    if self._checkpoint_future and not self._checkpoint_future.done():
      # Previous checkpoint needs to finish before saving the next one.
      wait_start = time.perf_counter()

      print_rank_n(
        f"Rank {self._rank}: previous checkpoint has not finished. "
        f"Checkpointing frequency is too high. Waiting...",
        rank=self._rank
      )

      self._checkpoint_future.result()

      print_rank_n(
        f"Rank {self._rank}: waited {time.perf_counter() - wait_start:.2f} "
        f"seconds for previous checkpoint to finish",
        rank=self._rank
      )
      self._checkpoint_future = None

    cp_start = time.perf_counter()

    if save_async:

      def callback(f: Future) -> None:
          if f.exception() is None:
            print_rank_n(
              f"Rank {self._rank}: Checkpoint is saved asynchronously "
              f"to {checkpoint_path} successfully.",
              rank=self._rank
            )
          else:
            print_rank_n(
              f"Rank {self._rank}: Checkpoint failed to save asynchronously to {checkpoint_path} "
              f"with the exception {f.exception()}",
              rank=self._rank
            )

      self._checkpoint_future = async_save(
        state_dict=state_dict,
        storage_writer=FileSystemWriter(
          checkpoint_path,
          thread_count=16
        ),
        process_group=self._process_group,
      )

      print_rank_n(
        f"Rank {self._rank}: Trainer was blocked for {time.perf_counter() - cp_start:.2f} seconds "
        "for checkpointing to finish...",
        rank=self._rank
      )

      self._checkpoint_future.add_done_callback(callback)
    else:
      print_rank_0(
        f"Saving model checkpoint synchronously to {checkpoint_path}.",
      )

      save(
          state_dict=state_dict,
          storage_writer=FileSystemWriter(
            checkpoint_path,
            thread_count=4
          ),
          process_group=self._process_group,
      )

    print_rank_0(
      "The full model checkpoint, including all the weights and "
      "configurations, has been saved successfully by the "
      "DistributedCheckpointer. "
      "You can now use this checkpoint for further training.",
    )

class AppState(Stateful):
  """This is a useful wrapper for checkpointing the Application State. 
     Since this object is compliant with the Stateful protocol, DCP will 
     automatically call state_dict/load_stat_dict as needed in the 
     dcp.save/load APIs.

  Note: We take advantage of this wrapper to hande calling distributed 
    state dict methods on the model and optimizer.
  """

  def __init__(self, model, optimizer=None, call_back=None):
    self.model = model
    self.call_back = call_back

  def set_call_back(self, cb):
    self.call_back = cb
    return self

  def state_dict(self):
    # this line automatically manages FSDP FQN's, as well as sets the 
    # default state dict type to FSDP.SHARDED_STATE_DICT
    model_state_dict = \
      get_model_state_dict(self.model)
    if self.call_back is not None:
      model_state_dict = self.call_back(model_state_dict)
    return {
      "model": model_state_dict
    }

  def load_state_dict(self, state_dict):
    # sets our state dicts on the model and optimizer, now that we've loaded
    set_model_state_dict(
      self.model,
      model_state_dict=state_dict["model"],
    )

class CheckpointConverter:
  def __init__(self, model_dir: Optional[str] = None):
    self.model_dir = model_dir

  def __call__(self, state_dict: STATE_DICT_TYPE) -> STATE_DICT_TYPE:
    return state_dict
