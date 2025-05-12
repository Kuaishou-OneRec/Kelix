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
from safetensors.torch import save_file
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
          if "packing" in key:
              # print(key)
              # print(f.get_tensor(key).shape)
              # print("================================================")
              continue
          if "vision_model" in key:
              pt1[key] = f.get_tensor(key)
  pt2 = {}
  for i in range(1, 6):
      with safe_open("/llm_reco_ssd/zhouyang12/models/Qwen3-8B-Base/model-0000" + str(i) + "-of-00005.safetensors", framework="pt", device="cpu") as f:
          for key in f.keys():
              pt2[key] = f.get_tensor(key)
  print(pt2.keys())
  print('lalallalalallal')
  for key in pt1.keys():
    keypt2 = "visual." + key
    pt2[keypt2] = pt1[key]
  #for key in pt2.keys():
    # print(key)
    # print(pt2[key].shape)
    # print("================================================")
  outputdir = "/llm_reco_ssd/zhouyang12/models/Qwen3-8B-Base-siglip_test"
  os.makedirs(outputdir, exist_ok=True)
  #merge pt1 and pt2
  save_file(pt2, outputdir + "/model.safetensors",metadata={"format": "pt"})
  pt3 = {}
  with safe_open(outputdir + "/model.safetensors", framework="pt", device="cpu") as f:
    for key in f.keys():
      pt3[key] = f.get_tensor(key)
  closecnt =0 
  for key in pt3.keys():
    if key in pt2.keys():
      #check tensor allclose
      if not torch.allclose(pt3[key], pt2[key], atol=1e-7):
        print(key)
        print(pt3[key].shape)
        print(pt2[key].shape)
        print("================================================")
        closecnt += 1 
  print(closecnt)
  # for key in pt3.keys():
  #   print(key)
  #   print(pt3[key].shape)
  #   print("================================================")
if __name__ == "__main__":
  main()