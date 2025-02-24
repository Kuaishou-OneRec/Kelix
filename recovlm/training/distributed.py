# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
from itertools import chain
from typing import Any, Callable, cast, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn

from torch.distributed._composable.fsdp import CPUOffloadPolicy, fully_shard
from torch.distributed._tensor import distribute_tensor, DTensor
from torch.distributed._tensor.placement_types import DTensorSpec, TensorMeta
from torch.distributed.checkpoint.state_dict import (
    _init_optim_state,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import ShardingStrategy
from torch.nn.modules.module import _IncompatibleKeys
from torch.optim import Optimizer


torch_version = torch.__version__
# TODO: Fix issues with DSD before uncommenting. See #2313 and #2277.
# _DISTRIBUTED_STATE_DICT_API_IS_AVAILABLE = (
#     "dev" not in torch_version and torch_version_ge("2.6.0")
# ) or ("dev" in torch_version and torch_version.split("dev")[1] >= "20241220")
_DISTRIBUTED_STATE_DICT_API_IS_AVAILABLE = False


def is_distributed() -> bool:
    """Check if all environment variables required to initialize torch.distributed are set
    and distributed is properly installed. This indicates a distributed run.
    https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization

    Checks the following conditions:

    * torch.distributed is available
    * master port and master address environment variables are set
    * world size is >1
    * rank environment variable is set

    Returns:
        bool: True if all of the above conditions hold, False otherwise.
    """
    port = os.environ.get("MASTER_PORT", "")
    addr = os.environ.get("MASTER_ADDR", "")
    size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", -1))
    avlb = dist.is_available()
    return bool(port and addr and size >= 1 and rank >= 0 and avlb)


def get_distributed_backend(device_type: str, offload_ops_to_cpu: bool = False) -> str:
    """Gets the PyTorch Distributed backend based on device type.

    Args:
        device_type (str): Device type to get backend for.
        offload_ops_to_cpu (bool, optional): Flag to check if any operations should be offloaded to CPU.
            Examples of these kinds of operations are CPU offload for FSDP and asynchronous save for distributed
            checkpointing. Defaults to False.

    Example:
        >>> get_distributed_backend("cuda")
        'nccl'
        >>> get_distributed_backend("cpu")
        'gloo'
        >>> get_distributed_backend("cuda", offload_ops_to_cpu=True)
        'cuda:nccl,cpu:gloo'

    Returns:
        str: Distributed backend for use in ``torch.distributed.init_process_group``.
    """
    default_device_backend_map = dist.Backend.default_device_backend_map
    backend = "nccl"
    if device_type in default_device_backend_map:
        backend = default_device_backend_map[device_type]
    if offload_ops_to_cpu:
        backend = f"{device_type}:{backend},cpu:gloo"
    return backend


def validate_no_params_on_meta_device(model: nn.Module) -> None:
    """
    Utility to validate that model has no params or buffers on meta device.
    If a meta param or buffer is found, an error indicating the param name will
    be raised.

    Args:
        model (nn.Module): model to check for meta params

    Raises:
        RuntimeError: If meta params or buffers exist in model
    """
    for n, p in chain(model.named_parameters(), model.named_buffers()):
        if p.is_meta:
            raise RuntimeError(f"Unexpected param or buffer {n} on meta device.")

def get_shard_conditions(
    name: str,
    module: nn.Module,
    names_to_match: Optional[List[str]] = None,
    *args,
    **kwargs,
) -> bool:
    """
    Returs True for layers named {}.layers.i or layers that exactly match names_to_match, otherwise,
    returns False. This is a helper function for sharding a model with FSDP.
    In :func:`~torchtune.training.shard_model`, we iterate over the model's named modules
    and apply fully_shard using this condition.

    As part of our sharding strategy, we want each layer to be sharded separately, as this is
    generally efficient. We may also want to shard certain modules that are not layers, such as
    the embedding module.

    #TODO: a more robust way would be to shard on the module type, not the name.

    Args:
        name (str): Name of the module.
        module (nn.Module): Module to be sharded.
        names_to_match (Optional[List[str]]): List of names to match, if any.
        *args: Variable length argument list to be passed to the Embedding module.
        **kwargs: Arbitrary keyword arguments to be passed to the Embedding module.

    Returns:
        bool: True if the module name matches the condition, False otherwise.

    Examples:
        >>> names_to_match = ["embedding"]
        >>> layer_names = ["layers.0", "decoder.layers.1", "encoder.layers.2.attention",
            "my_wrapper.layer.1.something", "embedding"]
        >>> matches = []
        >>> for name in layer_names:
        >>>     if shard_condition_is_layer_or_match(name, None): matches.append(name)
        >>> print(matches)
        >>> ["layers.0", "decoder.layers.1", "embedding"]
    """
    if names_to_match and name in names_to_match:
        return True

    name_list = name.split(".")
    if len(name_list) >= 2:
        return name_list[-2] == "layers" and str.isdigit(name_list[-1])

    return False


def shard_model(
    model: nn.Module,
    shard_conditions: List[Callable[[str, nn.Module], bool]],
    *,
    cpu_offload: bool,
    reshard_after_forward: bool = True,
    dp_mesh: Optional[DeviceMesh] = None) -> None:
    """
    Utility to shard a model with FSDP using the PyTorch Distributed fully_shard API.

    This method will over the model's named modules from the bottom-up and apply shard modules
    based on whether they meet any of the criteria from shard_conditions.

    Args:
        model (TransformerDecoder): Model to shard with FSDP.
        shard_conditions (List[Callable[[str, nn.Module], bool]]): A list of functions to determine
            which modules to shard with FSDP. Each function should take module name (relative to root)
            and the module itself, returning True if FSDP should shard the module and False otherwise.
            If any of shard_conditions return True for a given module, it will be sharded by FSDP.
        cpu_offload (bool): If set to True, FSDP will offload parameters, gradients, and optimizer
            states to CPU.
        reshard_after_forward (bool): Whether to reshard parameters and buffers after
            the forward pass. Setting this to True corresponds to the FULL_SHARD sharding strategy
            from FSDP1, while setting it to False corresponds to the SHARD_GRAD_OP sharding strategy.
        dp_mesh (Optional[DeviceMesh]): Device mesh to use for FSDP sharding under mutliple parallelism.
            Default to None.

    Raises:
        ValueError: If no layer modules were sharded, indicating that no shard_condition was triggered.
    """
    fsdp_kwargs = {"reshard_after_forward": reshard_after_forward, "mesh": dp_mesh}
    if cpu_offload:
        fsdp_kwargs["offload_policy"] = CPUOffloadPolicy()

    # Shard the model with FSDP, iterating in reverse to start with
    # lowest-level modules first
    num_layers_sharded = 0
    for n, m in reversed(list(model.named_modules())):
        if any([shard_condition(n, m) for shard_condition in shard_conditions]):
            fully_shard(m, **fsdp_kwargs)
            num_layers_sharded += 1

    if num_layers_sharded == 0:
        raise ValueError(
            "No layer modules were sharded. Please check if shard conditions are working as expected."
        )

    # Finally shard the entire model to account for any stragglers
    fully_shard(model, **fsdp_kwargs)

def load_from_full_model_state_dict(model: "FSDPModule", full_sd: Dict[str, Any]):
    meta_sharded_sd = model.state_dict()
    sharded_sd = {}
    if dist.get_rank() == 0:
        assert len(meta_sharded_sd) == len(full_sd), \
            "Sharded State Dict doesn't equal to Full State Dict"
        assert sorted(list(meta_sharded_sd.keys())) == sorted(list(full_sd.keys())), \
            "Keys of Sharded State Dict doesn't equal to Full State Dict"
        # for param_name, full_param in full_sd.items():
        #     sharded_meta_param = meta_sharded_sd[param_name]
        #     full_param = full_param.detach().cuda()
        #     mesh = sharded_meta_param.device_mesh
        #     dist.broadcast(full_param, src=0, group=mesh.get_group(0))
        #     sharded_tensor = distribute_tensor(
        #         full_param, mesh, sharded_meta_param.placements
        #     )
        #     print(f"Load: {param_name}, {full_param.shape}, {type(full_param)}, {sharded_meta_param.shape},  {type(sharded_meta_param)}")
        #     sharded_sd[param_name] = nn.Parameter(sharded_tensor)
    for param_name, sharded_meta_param in meta_sharded_sd.items():
        if dist.get_rank() == 0:
            full_tensor = full_sd[param_name].detach().cuda()
        else:
            full_tensor = torch.empty(
                sharded_meta_param.size(),
                device="cuda",
                dtype=sharded_meta_param.dtype,
            )
        print(f"before {param_name}, {full_tensor.shape}, {type(full_tensor)}, {sharded_meta_param.shape},  {type(sharded_meta_param)}")
        mesh = sharded_meta_param.device_mesh
        dist.broadcast(full_tensor, src=0, group=mesh.get_group(0))
        sharded_tensor = distribute_tensor(
            full_tensor, mesh, sharded_meta_param.placements
        )
        sharded_sd[param_name] = nn.Parameter(sharded_tensor)
        if dist.get_rank() == 0:
            print(f"Load & redistribute: {param_name}")
        
    model.load_state_dict(sharded_sd, assign=True)
