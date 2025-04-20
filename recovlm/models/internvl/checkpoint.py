from typing import Dict, Any
import tqdm
import re
import torch
from recovlm.training.checkpoint import CheckpointConverter
from recovlm.models.internvl.configuration_internvl_chat import Qwen2VLConfig


class InternVLCheckpointConverter:
  def __init__(self, model_path_or_name: str = None):
    self.model_path_or_name = model_path_or_name

  def __call__(self, state_dict):
     return self.convert(state_dict)

  def convert(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if 'language_model.model.embed_tokens.weight' not in state_dict:
        assert 'language_model.lm_head.weight' in state_dict
        state_dict['language_model.model.embed_tokens.weight'] = state_dict['language_model.lm_head.weight'].clone()
        print(f"[INFO] Added missing embed_tokens.weight to state_dict ({state_dict['language_model.model.embed_tokens.weight'].shape})")
        return state_dict
    elif 'language_model.lm_head.weight' not in state_dict:
        state_dict['language_model.lm_head.weight'] = state_dict['language_model.model.embed_tokens.weight'].clone()
        print(f"[INFO] Added missing lm_head.weight to state_dict ({state_dict['language_model.lm_head.weight'].shape})")
        return state_dict
    return state_dict

  def revert(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    pass



if __name__ == "__main__":
    # _test_convert()
    pass


