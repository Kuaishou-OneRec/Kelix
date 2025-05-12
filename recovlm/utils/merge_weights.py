import argparse
import re
import os
from recovlm.models.qwen_3_vl_2.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration_siglip
from recovlm.models.qwen_3_vl_2.configuration_qwen2_5_vl import Qwen2_5_VLConfig
from recovlm.models.qwen_3_vl_2.modeling_qwen2_5_vl import Qwen2_5_VLModel
from recipes.ViT.training.models.siglip.modeling_siglip import SiglipVisionModel
import torch
import transformers
# Qwen2VLForConditionalGeneration


def get_argument_parser():
  parser = argparse.ArgumentParser()

  parser.add_argument("--model_dir", type=str, default="/llm_reco_ssd/zhouyang12/models/Qwen3-8B-Base",
                      help="The directory of the pretrained LLM.")

  parser.add_argument("--vision_encoder_dir", type=str, default="/llm_reco/liuyang76/Models/siglip2-so400m-patch14-384",
                      help="The directory of the pretrained ViT.")

  parser.add_argument("--new_model_dir", type=str, default="/llm_reco_ssd/zhouyang12/models/Qwen3-8B-Base-siglip",
                      help="The directory of the pretrained ViT.")

  return parser


def main():
  arg_parser = get_argument_parser()
  args = arg_parser.parse_args()

  # llm weights
  model_config = Qwen2_5_VLForConditionalGeneration_siglip.config_class.from_pretrained(
    args.new_model_dir)
  model = Qwen2_5_VLForConditionalGeneration_siglip(model_config)

  text_model = Qwen2_5_VLModel.from_pretrained(
    args.model_dir)

  sd = model.state_dict()
  text_sd = text_model.state_dict()

  for name in text_sd.keys():
    assert name in sd
    print(name)
    sd[name] = text_sd[name]

  if args.vision_encoder_dir:
    vision_encoder = SiglipVisionModel.from_pretrained(
        args.vision_encoder_dir)

    vision_sd = vision_encoder.vision_model.state_dict()
  for name in vision_sd.keys():
    assert name in sd
    print(name)
    sd[name] = vision_sd[name]

  model.load_state_dict(sd)
  if not os.path.exists(args.new_model_dir):
    os.makedirs(args.new_model_dir)
  model.save_pretrained(args.new_model_dir)


if __name__ == "__main__":
  main()