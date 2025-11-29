"""
Checkpoint Management for Distributed Training.

This module provides utilities for saving and loading checkpoints in distributed
training scenarios, with support for:

- SafeTensors and PyTorch binary formats
- Sharded checkpoints (HuggingFace style)
- FSDP (Fully Sharded Data Parallel) state management
- Asynchronous checkpoint saving
- Distributed checkpoint protocol (DCP)

The module handles both single-file and multi-file checkpoints, automatically
detecting and loading sharded weights. It supports checkpoint resumption and
provides utilities for gathering distributed model states.

Classes:
    DistributedCheckpointer: Main checkpointer for distributed training
    AppState: Stateful wrapper for model and optimizer
    CheckpointerInterface: Protocol defining checkpointer interface

Functions:
    load_safetensors: Load safetensors file
    safe_torch_load: Safely load checkpoint to CPU
    load_hf_checkpoint: Load HuggingFace-style sharded checkpoint
    gather_cpu_state_dict: Gather FSDP state dict to CPU
    get_latest_checkpoint_path: Get path to latest checkpoint
    get_checkpoint_path: Get checkpoint path by ID
    save_checkpoint: High-level checkpoint saving utility

Example:
    >>> # Create checkpointer
    >>> checkpointer = DistributedCheckpointer()
    >>> 
    >>> # Save checkpoint
    >>> app_state = AppState(model, optimizer)
    >>> save_checkpoint(app_state, checkpointer, "./checkpoints", global_step=1000)
    >>> 
    >>> # Load checkpoint
    >>> state_dict = {"app": app_state}
    >>> checkpointer.load_checkpoint(state_dict, "./checkpoints/global_step1000")
"""
from typing import Dict, Any, Union, Optional, List, Protocol
import collections
import re
import os
import gc
import glob
import time
import torch
from pathlib import Path
from safetensors import safe_open
import torch.distributed as dist
from concurrent.futures import Future

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
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

from muse.training.distributed import get_world_size_and_rank
from muse.utils.common import print_rank_0, print_rank_n

CHECKPOINT_ID_PREFIX = "global_step"

def load_safetensors(path: Union[str, Path]) -> Dict[str, torch.Tensor]:
    """
    Load tensors from a safetensors file.
    
    SafeTensors is a safe format for storing tensors that prevents code execution
    vulnerabilities present in pickle-based formats.
    
    Args:
        path (str or Path): Path to the .safetensors file
        
    Returns:
        Dict[str, torch.Tensor]: Dictionary mapping tensor names to tensors (on CPU)
        
    Example:
        >>> tensors = load_safetensors("model.safetensors")
        >>> print(tensors.keys())
        dict_keys(['model.layer1.weight', 'model.layer1.bias', ...])
    """
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    return tensors

def safe_torch_load(
    checkpoint_path: Union[Path, str],
    weights_only: bool = True,
    mmap: bool = True,
) -> Dict[str, Any]:
    """
    Utility to load a checkpoint file onto CPU in a safe manner. 
    Provides separate handling for safetensors files.

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
      raise ValueError(
        f"Unable to load checkpoint from {checkpoint_path}. ") from e
    return state_dict

def load_hf_checkpoint(model_dir: str) -> Dict[str, torch.Tensor]:
    """
    Load HuggingFace-style sharded checkpoint from a directory.
    
    Loads and merges multiple checkpoint shards into a single state dict. Supports
    both safetensors (.safetensors) and PyTorch binary (.bin) formats. Automatically
    detects all shard files in the directory.
    
    Args:
        model_dir (str): Directory containing checkpoint shards
        
    Returns:
        Dict[str, torch.Tensor]: Merged state dict with all model parameters
        
    Raises:
        ValueError: If state dict contains non-tensor values
        
    Note:
        - Checkpoint files are loaded sequentially to manage memory
        - Garbage collection is performed after each shard to free memory
        - Progress is logged on rank 0
        
    Example:
        >>> state_dict = load_hf_checkpoint("./model_shards/")
        >>> # Loads: model-00001-of-00004.safetensors, model-00002-of-00004.safetensors, etc.
        >>> print(f"Loaded {len(state_dict)} parameters")
    """
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
    """
    Protocol defining the interface for checkpoint managers.
    
    Classes implementing this protocol must provide load_checkpoint and
    save_checkpoint methods with the specified signatures.
    
    Methods:
        load_checkpoint: Load checkpoint from storage
        save_checkpoint: Save checkpoint to storage
    """
    
    def load_checkpoint(self, **kwargs) -> Dict[str, Any]:
        """Load checkpoint from storage."""
        ...

    def save_checkpoint(self, state_dict: Dict[str, Any], **kwargs) -> None:
        """Save checkpoint to storage."""
        ...

def get_latest_checkpoint_path(checkpoint_dir: str) -> Optional[str]:
    """
    Get the path to the latest checkpoint in a directory.
    
    Reads the "latest" file in the checkpoint directory to find the most recent
    checkpoint subdirectory.
    
    Args:
        checkpoint_dir (str): Root checkpoint directory
        
    Returns:
        str or None: Full path to latest checkpoint directory, or None if no
            "latest" file exists
            
    Example:
        >>> latest_path = get_latest_checkpoint_path("./checkpoints")
        >>> # Returns: "./checkpoints/global_step5000" if that's the latest
    """
    if os.path.exists(os.path.join(checkpoint_dir, "latest")):
        with open(os.path.join(checkpoint_dir, "latest"), "r") as f:
            latest_checkpoint_dir = f.read().strip()
        return os.path.join(checkpoint_dir, latest_checkpoint_dir)
    else:
        return None

def get_checkpoint_path(checkpoint_dir: str,
                        checkpoint_id: Optional[str] = None) -> Optional[str]:
    """
    Get checkpoint path by ID or latest.
    
    If checkpoint_id is provided, returns the path to that specific checkpoint.
    Otherwise, returns the latest checkpoint path.
    
    Args:
        checkpoint_dir (str): Root checkpoint directory
        checkpoint_id (str, optional): Specific checkpoint ID (e.g., "global_step1000").
            If None, returns latest checkpoint. Defaults to None.
            
    Returns:
        str or None: Full checkpoint path, or None if no checkpoint found
        
    Example:
        >>> # Get specific checkpoint
        >>> path = get_checkpoint_path("./ckpts", "global_step1000")
        >>> # Returns: "./ckpts/global_step1000"
        >>> 
        >>> # Get latest checkpoint
        >>> path = get_checkpoint_path("./ckpts")
        >>> # Returns: "./ckpts/global_step5000" (if that's the latest)
    """
    if checkpoint_id:
        return os.path.join(checkpoint_dir, checkpoint_id)
    else:
        return get_latest_checkpoint_path(checkpoint_dir)

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
    _, self._rank = get_world_size_and_rank()
    self._process_group: Optional[dist.ProcessGroup] = process_group

  def load_checkpoint(self,
                      state_dict: STATE_DICT_TYPE,
                      checkpoint_path: str) -> Dict[str, Any]:
    """
    Load a Distributed checkpoint saved at the <checkpoint_path>
    If no path is provided, latest intermediate checkpoint is loaded.
    """

    print_rank_0(f"Loading checkpoint from {checkpoint_path}")

    dcp.load(
      state_dict=state_dict,
      storage_reader=FileSystemReader(checkpoint_path),
      process_group=self._process_group,
    )

    return state_dict

  def save_checkpoint(
      self,
      state_dict: STATE_DICT_TYPE,
      checkpoint_path: str,
      save_async: bool = False) -> None:
    """
    Save a distributed checkpoint to storage.
    If ``save_async`` is True, the save happens asynchronously unblocking the GPUs sooner. This
    should only be used for the intermediate checkpoints. Final checkpoint has to be a synchronous
    one as the finetuning job can not terminate until the checkpoint gets persisted.

    Args:
      state_dict (Dict[str, Any]): Checkpoint state dict to be written out to file
      checkpoint_path (str): Path to save the checkpoint
      save_async (bool): If True, save the checkpoint asynchronously
    """
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
    """
    Stateful wrapper for model and optimizer checkpoint management.
    
    This class implements the Stateful protocol, allowing DCP (Distributed
    Checkpoint) to automatically manage state dict operations. It handles:
    - FSDP fully qualified names (FQNs)
    - Sharded state dict types
    - Model and optimizer state coordination
    
    The wrapper simplifies checkpoint saving/loading by automatically calling
    the appropriate distributed state dict methods.
    
    Args:
        model (nn.Module): Model to checkpoint (may be FSDP-wrapped)
        optimizer (Optimizer, optional): Optimizer to checkpoint. Defaults to None.
        
    Example:
        >>> model = MyModel()
        >>> optimizer = torch.optim.Adam(model.parameters())
        >>> app_state = AppState(model, optimizer)
        >>> 
        >>> # Save
        >>> state_dict = {"app": app_state}
        >>> dcp.save(state_dict, checkpoint_path=path)
        >>> 
        >>> # Load
        >>> new_app_state = AppState(new_model, new_optimizer)
        >>> state_dict = {"app": new_app_state}
        >>> dcp.load(state_dict, checkpoint_path=path)
    """

    def __init__(self, model, optimizer=None):
        """
        Initialize AppState with model and optional optimizer.
        
        Args:
            model (nn.Module): Model to manage
            optimizer (Optimizer, optional): Optimizer to manage
        """
        self.model = model
        self.optimizer = optimizer

    def state_dict(self) -> Dict[str, Any]:
        """
        Get distributed state dict for model and optimizer.
        
        Automatically manages FSDP FQNs and sets the default state dict type
        to FSDP.SHARDED_STATE_DICT for efficient distributed checkpointing.
        
        Returns:
            Dict[str, Any]: State dict containing "model" and "optim" keys
        """
        # this line automatically manages FSDP FQN's, as well as sets the 
        # default state dict type to FSDP.SHARDED_STATE_DICT
        model_state_dict, optimizer_state_dict = \
            get_state_dict(self.model, self.optimizer)

        return {
            "model": model_state_dict,
            "optim": optimizer_state_dict
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        Load state dict into model and optimizer.
        
        Sets the loaded state dicts on the model and optimizer using distributed
        state dict APIs for proper FSDP handling.
        
        Args:
            state_dict (Dict[str, Any]): State dict with "model" and "optim" keys
        """
        # sets our state dicts on the model and optimizer, now that we've loaded
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optimizer_state_dict=state_dict["optim"],
        )


# TODO: support saving dataloader, lr_scheduler, training states, etc.
def save_checkpoint(
    app_state: AppState,
    dist_checkpointer: DistributedCheckpointer,
    checkpoint_dir: str,
    global_step: int = None
  ) -> None:
    """Utility function to checkpoint tranining states.

    Args:
        app_state: The app state to save
        dist_checkpointer: The dist checkpointer to save
        checkpoint_dir: The directory to save the checkpoint
        global_step: The global step to save the checkpoint
    """
    if dist.get_rank() == 0:
      os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_id = f"{CHECKPOINT_ID_PREFIX}{global_step}"

    checkpoint_path = get_checkpoint_path(
      checkpoint_dir, checkpoint_id)

    if dist.get_rank() == 0:
      os.makedirs(checkpoint_path, exist_ok=True)

      # Update latest file
      with open(os.path.join(checkpoint_dir, "latest"), "w") as f:
        f.write(checkpoint_id + "\n")
    
    dist.barrier()

    dist_checkpointer.save_checkpoint(
      state_dict={"app": app_state},
      checkpoint_path=checkpoint_path
    )
    dist.barrier()
