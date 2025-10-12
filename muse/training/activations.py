import torch.nn as nn

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

def set_activation_checkpointing(
    model: nn.Module, auto_wrap_policy, **kwargs
) -> None:
    """Utility to apply activation checkpointing to the passed-in model.

    Args:
        model (nn.Module): Model to apply activation checkpointing to.
        auto_wrap_policy (ACWrapPolicyType): Policy to wrap module.
            This can either be a set of ``nn.Module`` types, in which case, modules of the specified type(s)
            will be wrapped individually with activation checkpointing, or a ``callable`` policy describing
            how to wrap the model with activation checkpointing. For more information on authoring custom
            policies, please see this tutorial:
            https://pytorch.org/tutorials/intermediate/FSDP_adavnced_tutorial.html#transformer-wrapping-policy.
        **kwargs: additional arguments to pass to ``torch.distributed`` activation checkpointing.
    """
    if isinstance(auto_wrap_policy, set):
        auto_wrap_policy = ModuleWrapPolicy(auto_wrap_policy)
    apply_activation_checkpointing(model, auto_wrap_policy=auto_wrap_policy, **kwargs)
