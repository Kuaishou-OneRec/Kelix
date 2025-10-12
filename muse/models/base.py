import torch
import torch.nn as nn

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
    
    def train(self):
        raise NotImplementedError(
            "Subclass must implement train method")
    
    def eval(self):
        raise NotImplementedError(
            "Subclass must implement eval method")
    
    @property
    def training(self):
        pass

    def get_checkpointing_modules(self):
        raise NotImplementedError(
            "Subclass must implement get_checkpointing_policy method")

        

# case1: pass model_config and random init (5%)
# case2: pass model_dir and random init (5%)
# case3: pass model_dir and init from pretrained (90%)