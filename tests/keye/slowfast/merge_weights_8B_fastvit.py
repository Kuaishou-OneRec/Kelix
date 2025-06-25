import argparse
import re
import os

import torch
import transformers
from safetensors import safe_open
from safetensors.torch import save_file
# Qwen2VLForConditionalGeneration

from transformers import AutoConfig
import json

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
  # Load the PyTorch model file

  fast_siglip_pt = {}
  with safe_open("/mmu_mllm_hdd_2/zhouyang12/output2/Keye/0.8.0/ViT/80m/0.0.1/global_step19800/vision_model.safetensors", framework="pt", device="cpu") as f:
    for key in f.keys():
        if key.startswith("visual"):
          cjx_key = key.replace("visual", "visual_fast")
          fast_siglip_pt[cjx_key] = f.get_tensor(key)

  

#   model_path = "/mmu_mllm_hdd_2/zangdunju/ckpt/global_step18200/vision_model.pth"
#   ptm = torch.load(model_path, map_location='cpu')
#   pt1 = {}
#   # Print the keys in the state dict
#   if isinstance(ptm, dict):
#       for key in ptm.keys():
#           if "visual" in key:
#               pt1[key] = ptm[key]

  # for key in pt1.keys():
  #   print(key)
  #   print(pt1[key].shape)
  #   print("================================================")
#   pt2 = {}
#   for i in range(1, 3):
#       with safe_open("/llm_reco_ssd/zhouyang12/models/Qwen3-1.7B/model-0000" + str(i) + "-of-00002.safetensors", framework="pt", device="cpu") as f:
#           for key in f.keys():
#               pt2[key] = f.get_tensor(key)
#   print('lalallalalallal')

  pt2 = {}
  for i in range(1, 3):
      with safe_open("/llm_reco_ssd/zhouyang12/models/Keye-8B-demo_hf_vit_rope/model.safetensors", framework="pt", device="cpu") as f:
          for key in f.keys():
              pt2[key] = f.get_tensor(key)
  print('lalallalalallal')
  
  import pdb
  pdb.set_trace()
  
  """
  
  """
#   for key in pt1.keys():
#     pt2[key] = pt1[key]
  for key in fast_siglip_pt.keys():
    pt2[key] = fast_siglip_pt[key]


  outputdir = "/llm_reco_ssd/zhouyang12/models/Keye-8B-demo_hf_vit_rope_slowfast_0625_fast_navit"
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
  for key in pt2.keys():
    assert key in pt3.keys()
    assert pt2[key].shape == pt3[key].shape
  for key in pt3.keys():
    if key in pt2.keys():
      continue
    else:
      print('not in pt2')
      print(key)
      print(pt3[key].shape)
      print("================================================")
  print('--------------------------------')
  print('--------------------------------')
  for key in ptm.keys():
    if key not in pt3.keys():
      print("not in pt3")
      print(key)
      print(ptm[key].shape)
      print("================================================")
  print("all close")


def config_print(model_name):
    config = AutoConfig.from_pretrained(model_name)
    print("=== Vision Config ===")
    print(json.dumps(config.vision_config.to_dict(), indent=2))




import torch
import os
import os.path as osp

def safetensors_to_bin(path):
    with safe_open(path, framework="pt", device="cpu") as fp:
        model = dict()
        for key in fp.keys():
            model[key] = fp.get_tensor(key)
    torch.save(model, osp.join(osp.dirname(path), "pytorch_model.bin"))


if __name__ == "__main__":
    # config_print("/llm_reco_ssd/zhouyang12/models/siglip-base-patch16-224")
    main()
    # safetensors_to_bin("/llm_reco_ssd/zhouyang12/models/Keye-2B-demo_fix_improc_slowfast/model.safetensors")