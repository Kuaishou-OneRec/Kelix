import argparse
import re
import os

import torch
import transformers
# Qwen2VLForConditionalGeneration


def get_argument_parser():
  parser = argparse.ArgumentParser()

  parser.add_argument("--llm_model_dir", type=str, default=None,
                      help="The directory of the pretrained LLM.")
  
  parser.add_argument("--vlm_model_dir", type=str, default=None,
                      help="The directory of the pretrained VLM.")
  
  parser.add_argument("--remove_params_path", type=str, default=None,
                      help="Remove params config.")

  parser.add_argument("--new_model_dir", type=str, default=None,
                      help="The directory of the pretrained ViT.")

  return parser


def main():
  arg_parser = get_argument_parser()
  args = arg_parser.parse_args()

  remove_params_set = set()
  if args.remove_params_path is not None:
    with open(args.remove_params_path) as fp:
      for line in fp:
        remove_params_set.add(line.strip())

  # llm weights
  model_config = transformers.Qwen2VLForConditionalGeneration.config_class.from_pretrained(
    args.new_model_dir)
  model = transformers.Qwen2VLForConditionalGeneration(model_config)
  sd = model.state_dict()
  loaded_params = []

  ############ load text model ####################
  text_model = transformers.AutoModelForCausalLM.from_pretrained(
    args.llm_model_dir)
  text_sd = text_model.state_dict()
  text_skip_params = []
  for name in text_sd.keys():
    assert name in sd
    if any(remove_name in name for remove_name in remove_params_set):
      text_skip_params.append(name)
      continue
    sd[name] = text_sd[name]
    loaded_params.append(name)
    print(f"load {name} from text_model")
  
  for name in text_skip_params:
    print(f"skip {name} from text_model")
  
  print(f"=" * 50)
  
  ############ load vision model ####################
  vision_model = transformers.Qwen2VLForConditionalGeneration.from_pretrained(args.vlm_model_dir)
  vision_sd = vision_model.state_dict()
  vision_skip_params = []
  for name in vision_sd.keys():
    if name.startswith("visual."):
      assert name in sd
      if any(remove_name in name for remove_name in remove_params_set):
        vision_skip_params.append(name)
        continue
      sd[name] = vision_sd[name]
      loaded_params.append(name)
      print(f"load {name} from vision_model")

  for name in vision_skip_params:
    print(f"skip {name} from vision_model")

  print(f"=" * 50)
  #################### dump models #################
  for name in loaded_params:
    print(f"loaded params: {name}")
  for name in sd:
    if name not in loaded_params:
      print(f"init params: {name}")

  print(f"total_load {len(loaded_params)} params")
  model.load_state_dict(sd)
  if not os.path.exists(args.new_model_dir):
    os.makedirs(args.new_model_dir)
  model.save_pretrained(args.new_model_dir)

if __name__ == "__main__":
  main()