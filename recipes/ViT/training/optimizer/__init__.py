import torch
import torch.nn as nn
from recipes.ViT.helpers.common import filter_function_arguments
from .muon import Muon


def build_optimizer(config, model, model_name):
    name = config.type
    if model_name != "siglip":
        raise NotImplementedError
    if name == "adamw":
        lr = config.learn_rate
        weight_decay = config.weight_decay
        betas = (config.beta1, config.beta2)
        return torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas
        )
    elif name == "muon":
        lr = config.learn_rate
        weight_decay = config.weight_decay
        muon_params = [
            p
            for name, p in model.named_parameters()
            if p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
        ]
        adamw_params = [
            p
            for name, p in model.named_parameters()
            if not (
                p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
            )
        ]

        return Muon(
            lr=lr,
            wd=weight_decay,
            muon_params=muon_params,
            adamw_params=adamw_params,
        )
    else:
        raise ValueError("optimizer not supported")
