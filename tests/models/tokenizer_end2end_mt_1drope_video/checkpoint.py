from typing import Dict, Any
import tqdm
import re
import torch
from recovlm.training.checkpoint import CheckpointConverter


class KeyeImageTokenizer_end2end_mt_1drope_videoCheckpointConverter(CheckpointConverter):
  def __init__(self, model_path_or_name: str = None, config=None):
    self.model_path_or_name = model_path_or_name
    self.config = config

  def __call__(self, state_dict, **kargs):
     return self.convert(state_dict)

  def convert(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    config = self.config
    n_q_tokens = config.vision_config.n_q_tokens
    embedding_dim = config.vision_config.embedding_dim
    total_dim = embedding_dim * n_q_tokens
    if 'model.embed_tokens.weight' not in state_dict:
        assert 'lm_head.weight' in state_dict
        state_dict['model.embed_tokens.weight'] = state_dict['lm_head.weight'].clone()
        print(f"[INFO] Added missing embed_tokens.weight to state_dict ({state_dict['model.embed_tokens.weight'].shape})")
    elif 'lm_head.weight' not in state_dict:
        state_dict['lm_head.weight'] = state_dict['model.embed_tokens.weight'].clone()
        print(f"[INFO] Added missing lm_head.weight to state_dict ({state_dict['lm_head.weight'].shape})")

    if getattr(self.config, "ar_mode", "none") == "ar":
        vocab_size = self.config.vocab_size 
        codebook_size = self.config.vision_config.codebook_size
        if state_dict['lm_head.weight'].shape[0] == vocab_size:
          old_weight = state_dict['lm_head.weight']
          new_weight = torch.nn.Linear(self.config.hidden_size, codebook_size + vocab_size, bias=False).weight
          with torch.no_grad(): 
            #  new_std, new_mean = new_weight.std(), new_weight.mean()
            #  old_std, old_mean = old_weight.std(), old_weight.mean()
             adjusted_new_weight = new_weight * 0.01 # (new_weight - new_mean) / new_std * old_std + old_mean
             adjusted_new_weight[:vocab_size].copy_(old_weight)
          print(f"[INFO] create new lm_head.weight with shape {new_weight.shape} from {old_weight.shape}, vocab_size={vocab_size}, codebook_size={codebook_size}")
          state_dict['lm_head.weight'] = adjusted_new_weight.to(old_weight)
        else:
           assert state_dict['lm_head.weight'].shape[0] == codebook_size + vocab_size, \
              f"lm_head.weight shape {state_dict['lm_head.weight'].shape} does not match codebook_size + vocab_size {codebook_size + vocab_size}"
    # if "token_head.weight" not in state_dict:
    #     n_q_tokens = config.vision_config.n_q_tokens
    #     hidden_size = config.hidden_size
    #     # torch.Size([8192, 1024])

    #     with torch.no_grad():
    #       state_dict["token_head.weight"] = torch.eye(hidden_size).repeat(n_q_tokens, 1)

    #       add_weight = torch.nn.Linear(hidden_size, n_q_tokens * hidden_size, bias=False).weight
    #       torch.nn.init.kaiming_normal_(add_weight, a=0, mode='fan_in', nonlinearity='relu')
    #       state_dict["token_head.weight"][hidden_size:] += add_weight[hidden_size:]
    #     # state_dict["token_head.weight"] = state_dict["token_head.weight"] * 0.1
    #     # print(f"[INFO] Added missing token_head.weight to state_dict ({state_dict['token_head.weight'].shape})")
    #     # print(state_dict["token_head.weight"])
    if "token_head.weight" in state_dict:
        del state_dict["token_head.weight"]
        print(f"[INFO] Removed token_head.weight from state_dict")

    return state_dict

  def revert(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return state_dict