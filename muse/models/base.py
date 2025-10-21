import torch
import torch.nn as nn
from typing import Callable
from functools import partial

class Model(nn.Module):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
    
    def forward(self, *args, **kwargs):
        pass

    def from_pretrained(self,
                        model_dir,
                        load_weights=True,
                        allow_random_init_params=None,
                        random_init_ratio=0.05,
                        ):
        raise NotImplementedError(
            "Subclass must implement from_pretrained method")

    def init_weights(self):
        raise NotImplementedError(
            "Subclass must implement init_weights method")

    def get_initializer(self, name: str) -> Callable[[torch.Tensor], None]:
        """Return a initializer function for the given name
        Args:
            name (str): The name of the parameter
        Returns:
            A callable function that takes a tensor(parameter weight) and initializes it,
            e.g,
                return partial(
                    nn.init.kaiming_normal_, 
                    a=0, mode='fan_in', nonlinearity='relu'
                )
        Examples:
            >>> model.get_initializer("layers.0.attention.q_proj.weight")(q_proj.weight)
        """
        raise NotImplementedError(
            "Subclass must implement get_initializer method")
    
    def train(self):
        """Set the model to training mode"""
        raise NotImplementedError(
            "Subclass must implement train method")
    
    def eval(self):
        """Set the model to evaluation mode"""
        raise NotImplementedError(
            "Subclass must implement eval method")
    
    @property
    def training(self):
        """Return whether the model is in training mode"""
        return self.training

    def get_checkpointable_module_classes(self):
        """Return a list of module classes that should be checkpointed"""
        raise NotImplementedError(
            "Subclass must implement get_checkpointable_module_classes method")

    def get_layers_to_shard(self):
        """Return a list of layers that should be sharded"""
        raise NotImplementedError(
            "Subclass must implement get_layers_to_shard method")
