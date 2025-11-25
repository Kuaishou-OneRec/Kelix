import torch
import torch.nn as nn
from typing import Callable, Optional, List, Dict, Any
from functools import partial
from muse.config.model_config import ModelConfig

class Model(nn.Module):
    """Base class for all models."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
    
    def forward(self, *args, **kwargs):
        pass

    @classmethod
    def from_pretrained(cls,
                        model_dir: str,
                        load_weights: bool = True,
                        allow_random_init_params: Optional[str] = None,
                        **kwargs) -> "Model":
        """Load weights from a pretrained model.
        Args:
            model_dir (str): The directory to load the weights from.
            load_weights (bool): Whether to load the weights.
            allow_random_init_params (Optional[str]): The parameters to allow random initialization.
            **kwargs: Additional keyword arguments.
        """
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
    
    def train(self, mode: bool = True):
        """Set the model to training mode.
        
        This recursively sets all submodules to training mode.
        PyTorch's nn.Module.train() automatically handles recursion.
        
        Args:
            mode (bool): Whether to set training mode (True) or evaluation mode (False).
                Default: True.
        
        Returns:
            self: Returns self for method chaining.
        """
        # Call nn.Module.train() directly to recursively set all submodules
        nn.Module.train(self, mode)
        return self
    
    def eval(self):
        """Set the model to evaluation mode.
        
        This recursively sets all submodules to evaluation mode.
        PyTorch's nn.Module.eval() automatically handles recursion.
        
        Returns:
            self: Returns self for method chaining.
        """
        # Call nn.Module.eval() directly to recursively set all submodules
        nn.Module.eval(self)
        return self
    
    def get_checkpointable_module_classes(self):
        """Return a list of module classes that should be checkpointed"""
        raise NotImplementedError(
            "Subclass must implement get_checkpointable_module_classes method")

    def get_layers_to_shard(self):
        """Return a list of layers that should be sharded by FSDP"""
        raise NotImplementedError(
            "Subclass must implement get_layers_to_shard method")

    def rope_init(self):
        """Initialize the RoPE for the model if needed"""
        pass

    def get_optimizer_grouped_parameters(
            self,
            learning_rate: float,
            weight_decay: float) -> List[Dict[str, Any]]:
        """Get the optimizer grouped parameters for AdamW optimizer
        Args:
            learning_rate (float): The learning rate.
            weight_decay (float): The weight decay.
        Returns:
            A list of optimizer grouped parameters for AdamW optimizer.
        """
        optimizer_grouped_parameters = [
          {
            "params": [p for n, p in self.named_parameters() 
                       if p.requires_grad],
            "weight_decay": weight_decay,
            "lr": learning_rate,
          },
        ]
    
    @classmethod
    def convert_hf_state_dict(cls,
                              hf_state_dict: Dict[str, torch.Tensor],
                              **kwargs) -> Dict[str, torch.Tensor]:
        """Convert a Hugging Face state dictionary to a model state dictionary
        Args:
            hf_state_dict (Dict[str, torch.Tensor]): The Hugging Face state dictionary.
            **kwargs: Additional keyword arguments.
        Returns:
            A dictionary of model state.
        """
        return hf_state_dict

    @classmethod
    def get_hf_state_dict(cls,
                          state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Convert the model state dict to the Hugging Face format"""
        return state_dict
    
    def generate(self, *args, **kwargs):
        """Generate text from the model"""
        raise NotImplementedError(
            "Subclass must implement generate method")
