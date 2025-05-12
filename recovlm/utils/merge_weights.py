import argparse
import re
import os
from recovlm.models.qwen_3_vl_2.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration_siglip
from recovlm.models.qwen_3_vl_2.configuration_qwen2_5_vl import Qwen2_5_VLConfig
from recovlm.models.qwen_3_vl_2.modeling_qwen2_5_vl import Qwen2_5_VLModel
from recipes.ViT.training.models.siglip.modeling_siglip import SiglipVisionModel
import torch
import transformers
from safetensors import safe_open
# Qwen2VLForConditionalGeneration


# def get_argument_parser():
#   parser = argparse.ArgumentParser()

#   parser.add_argument("--model_dir", type=str, default="/llm_reco_ssd/zhouyang12/models/Qwen3-8B-Base",
#                       help="The directory of the pretrained LLM.")

#   parser.add_argument("--vision_encoder_dir", type=str, default="/llm_reco/liuyang76/Models/siglip2-so400m-patch14-384",
#                       help="The directory of the pretrained ViT.")

#   parser.add_argument("--new_model_dir", type=str, default="/llm_reco_ssd/zhouyang12/models/Qwen3-8B-Base-siglip",
#                       help="The directory of the pretrained ViT.")

#   return parser


def main():
  # Load the safetensors file properly
  pt1 = {}
  with safe_open("/llm_reco/liuyang76/Models/siglip2-so400m-patch14-384/model.safetensors", framework="pt", device="cpu") as f:
      for key in f.keys():
          print(key)
          #pt1[key] = f.get_tensor(key)
  
  #pt2 = torch.load("/llm_reco_ssd/zhouyang12/models/Qwen3-8B-Base/model-00001-of-00005.safetensors")
  #merge pt1 and pt2

  #save the merged weights
  #torch.save(pt2, "/llm_reco_ssd/zhouyang12/models/Qwen3-8B-Base-siglip/model.safetensors")

if __name__ == "__main__":
  main()