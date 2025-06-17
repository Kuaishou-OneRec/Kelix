import argparse
import re
import os
import glob
import tqdm
import torch
from typing import Union, Dict

import os
import json
import argparse
import torch
from pathlib import Path
from safetensors.torch import save_file
import transformers
from safetensors import safe_open
from safetensors.torch import save_file
# Qwen2VLForConditionalGeneration
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict
from torch.distributed.checkpoint.metadata import Metadata, STATE_DICT_TYPE
from torch.distributed.checkpoint.default_planner import (
    _EmptyStateDictLoadPlanner
)

# /llm_reco_ssd/zhouyang12/models/siglip2-so400m-patch16-naflex
# /llm_reco_ssd/zangdunju/output2/RecoVLM/SigLIP/siglip_navit/global_step1000/model_float32.pth

def get_argument_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-32B",
                      help="model name (not used)")
  parser.add_argument("--model_dir", type=str, default="/mmu_mllm_hdd_2/lingzhixin/models/Qwen3-32B",
                      help="The directory of the pretrained LLM.")
  parser.add_argument("--delete_mlp", type=int, default=1,
                      help="whether delete mlp params")
  parser.add_argument("--delete_layer_mlp", type=int, default=0,
                      help="whether delete model mlp params. Usually used when target ckpt is MOE")
  # delete_layer_mlp
  # parser.add_argument("--vision_encoder_dir", type=str, default="/llm_reco_ssd/zangdunju/output2/RecoVLM/SigLIP/siglip_navit/global_step1000/model_float32.pth", #"/llm_reco_ssd/zhouyang12/models/siglip2-so400m-patch16-naflex",
  #                     help="The directory of the pretrained ViT.")

  parser.add_argument("--new_model_dir", type=str, default="/mmu_mllm_hdd_2/lingzhixin/models/Keye-32B-scratch_0606",
                      help="The directory of the pretrained ViT.")
  parser.add_argument("--vit_model_path", type=str, default="/llm_reco_ssd/zangdunju/output2/RecoVLM/SigLIP/siglip_navit/global_step1000/model_float32.pth", #"/mmu_mllm_hdd_2/zangdunju/ckpt/global_step18200/vision_model.pth",
                      help="The directory of the pretrained ViT.")
  return parser

"""
python3 /llm_reco/lingzhixin/recovlm_qw0510/recovlm/tools/model_helpers/hf_converters/convert_qwen3_to_keye.py --delete_mlp 0 --model_dir /llm_reco_ssd/zhouyang12/models/Keye-8B-demo_hf_vit_rope_slowfast_0608/ --vit_model_path /llm_reco_ssd/zhouyang12/models/Keye-8B-demo_hf_vit_rope_slowfast_0608/ --new_model_dir /llm_reco_ssd/zhouyang12/models/Keye-8B-demo_hf_vit_rope_slowfast_0608_split
"""
args = get_argument_parser().parse_args()


def snapshot_downloader():
  from huggingface_hub import snapshot_download
  model_name = args.model_name
  local_dir = args.model_dir # "本地保存路径"  # 可选，默认保存到 ~/.cache/huggingface/hub
  # 下载模型
  snapshot_download(
      repo_id=model_name,
      local_dir=local_dir,
      # revision="版本号",  # 可选，指定分支或提交哈希
      # ignore_files=[".gitattributes"],  # 可选，忽略某些文件
      resume_download=True,  # 可选，启用断点续传
  )



def demo():
  from transformers import AutoModelForCausalLM, AutoTokenizer
  model_name = args.new_model_dir
  model_name = args.model_dir
  # load the tokenizer and the model
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForCausalLM.from_pretrained(
      model_name,
      torch_dtype="auto",
      device_map="auto"
  )

  # prepare the model input
  prompt = "Give me a short introduction to large language models."
  messages = [
      {"role": "user", "content": prompt}
  ]
  text = tokenizer.apply_chat_template(
      messages,
      tokenize=False,
      add_generation_prompt=True,
      enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
  )
  model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
  model_inputs['input_ids'] = model_inputs['input_ids'] * 0
  print(model_inputs)
  # conduct text completion
  generated_ids = model.generate(
      **model_inputs,
      max_new_tokens=32768
  )
  output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

  # the result will begin with thinking content in <think></think> tags, followed by the actual response
  print(tokenizer.decode(output_ids, skip_special_tokens=True))


SHARD_FNAME = "model-{cpt_idx}-of-{num_shards}"

def dcp_to_torch_save(sd,
                      output_dir,
                      model_only: bool=True,
                      use_safetensor: bool=True,
                      max_gb_per_shard: int = 4,
                      model_type:str="Intern"):

  for k in tqdm.tqdm(sd):
    sd[k] = sd[k].to(torch.bfloat16)
  split_state_dicts: Dict[int, Dict[str, torch.Tensor]] = {}
  for key, value in tqdm.tqdm(sd.items()):
    split_state_dicts[key] = value
  
  split_state_dicts: Dict[int, Dict[str, torch.Tensor]] = {}
  cpt_idx = 0
  total_size = 0
  current_size = 0
  for key, weight in  tqdm.tqdm(sd.items()):
    if cpt_idx not in split_state_dicts:
      split_state_dicts[cpt_idx] = {}
    split_state_dicts[cpt_idx].update({key: weight})
    current_size += weight.numel() * weight.element_size()
    total_size += current_size
    if current_size >= max_gb_per_shard * 1024 * 1024 * 1024:
      cpt_idx += 1
      current_size = 0

  # write the partitioned state dicts to the right checkpoint file
  # e.g. model-00001-of-00004.safetensors, model-00002-of-00004.safetensors, etc
  num_shards = len(split_state_dicts)
  weight_map = {}
  for cpt_idx, model_state_dict in tqdm.tqdm(split_state_dicts.items()):
    # TODO: We should probably use the original shard name and just add a prefix
    # however, having the SHARD_FNAME standardizes our checkpoints
    shard_name = SHARD_FNAME.format(
      cpt_idx=f"{cpt_idx}".zfill(5), num_shards=f"{num_shards}".zfill(5)
    )
    output_path = Path(output_dir) / shard_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not use_safetensor:
      output_path = output_path.with_suffix(".bin")
      torch.save(model_state_dict, output_path)
    else:
      output_path = output_path.with_suffix(".safetensors")
      save_file(model_state_dict, output_path, metadata={"format": "pt"})
    for key, weight in model_state_dict.items():
      weight_map[key] = str(output_path.parts[-1])

    print(
      "Model checkpoint of size "
      f"{os.path.getsize(output_path) / 1024**3:.2f} GiB "
      f"saved to {output_path}"
    )
    
  if use_safetensor:
    weight_map_path = Path(output_dir) / "model.safetensors.index.json"
  else:
    weight_map_path = Path(output_dir) / "model.bin.index.json"
  with open(weight_map_path, "w") as f:
    f.write(json.dumps({
      "metadata": {
        "total_size": total_size
      },
      "weight_map": weight_map,
    }, indent=2))



import os
import glob
from concurrent.futures import ThreadPoolExecutor
import tqdm
# from safetensors.torch import safe_open
def load_safe_tensors(path, max_workers=None):
    """
    多线程加载指定目录下的所有 safetensors 文件
    
    参数:
        path: 要搜索的目录路径
        max_workers: 最大线程数，默认使用 CPU 核心数
    
    返回:
        包含所有张量的字典
    """
    # 获取所有 safetensors 文件路径
    file_paths = glob.glob(os.path.join(path, "*.safetensors"))
    
    # 初始化结果字典
    result = {}
    
    # 定义单个文件的加载函数
    def load_file(file_path):
        tensors = {}
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
        return tensors
    
    # 使用线程池并行加载文件
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务并获取 future 对象
        futures = [executor.submit(load_file, file_path) for file_path in file_paths]
        
        # 使用 tqdm 显示进度
        for future in tqdm.tqdm(futures, desc="加载 safetensors 文件", total=len(file_paths)):
            # 合并结果
            result.update(future.result())
    
    return result


import re

def is_valid_model_pattern(s: str) -> bool:
    """
    检查输入字符串是否匹配model.layers.NUM.self_attn.X_proj.bias模式
    
    参数:
        s (str): 待检查的字符串
    
    返回:
        bool: 匹配返回True，否则返回False
    """
    pattern = r'^model\.layers\.\d+\.self_attn\.[A-Za-z]_proj\.bias$'
    return bool(re.match(pattern, s))

import re

def is_model_layer_mlp_format(s):
    """
    判断字符串是否符合 model.layers.NUM.mlp.XXX 格式
    
    参数:
        s: 待检查的字符串
    
    返回:
        bool: 如果符合格式返回 True，否则返回 False
    """
    pattern = r'^model\.layers\.\d+\.mlp*'
    return bool(re.match(pattern, s))



def main():
  # Load the PyTorch model file
  vit_model_path = args.vit_model_path

  vit_sd = torch.load(vit_model_path, map_location='cpu') if vit_model_path.endswith(".pth") else load_safe_tensors(vit_model_path)

  llm_sd = load_safe_tensors(args.model_dir)

  vllm_sd = vit_sd
  for key in llm_sd.keys():
    vllm_sd[key] = llm_sd[key]

  outputdir = args.new_model_dir
  if args.delete_mlp:
    for k in 'mlp_AR.pre_norm.weight,mlp_AR.pre_norm.bias,mlp_AR.linear_1.weight,mlp_AR.linear_1.bias,mlp_AR.linear_2.weight,mlp_AR.linear_2.bias'.split(','):
      if k in vllm_sd:
        del vllm_sd[k]

  prj_bias = 0
  for k in list(vllm_sd.keys()):
    if not is_valid_model_pattern(k): continue
    if args.delete_layer_mlp and is_model_layer_mlp_format(k): continue
      
    print(f"{k} is deleted")
    if k in vllm_sd:
      del vllm_sd[k]
    prj_bias += 1
  
  # print(f"totally {prj_bias} bias killed")

  dcp_to_torch_save(vllm_sd, outputdir)

  reread_sd = load_safe_tensors(outputdir)


  assert len(reread_sd.keys()) == len(set(vit_sd.keys()) | set(llm_sd.keys())) == len(vllm_sd)
  for k, v in tqdm.tqdm(vllm_sd.items()):
    v2 = reread_sd[k]
    assert v2.shape == v.shape and torch.allclose(v, v2), f"bad with {k}, shape: {v2.shape}/{v.shape}"

if __name__ == "__main__":
  # PYTHONPATH=. python3 /llm_reco/lingzhixin/recovlm_qw0510/recovlm/tools/model_helpers/hf_converters/convert_qwen3_to_keye.py


  # PYTHONPATH=. python3 /llm_reco/lingzhixin/recovlm_qw0510/recovlm/tools/model_helpers/hf_converters/convert_qwen3_to_keye.py --vit_model_path /mmu_mllm_hdd_2/zhouyang12/output/Keye/Stage3_0.3.4_1pes_2e-5_resume3k/0.8.0/8b/step3000/global_step3000/converted_hf --new_model_dir /mmu_mllm_hdd_2/lingzhixin/models/Keye-32B-vit0.8.1_0606
  # PYTHONPATH=. python3 -m  versions.v0_8_1.Keye-32B.tests  /mmu_mllm_hdd_2/lingzhixin/models/Keye-32B-vit0.8.1_0606
  # delete_layer_mlp
  # PYTHONPATH=. python3 /llm_reco/lingzhixin/recovlm_qw0510/recovlm/tools/model_helpers/hf_converters/convert_qwen3_to_keye.py --vit_model_path /llm_reco_ssd/zangdunju/output2/RecoVLM/SigLIP/siglip_navit/global_step1000/model_float32.pth --model_dir /mmu_mllm_hdd_2/lingzhixin/models/Qwen3-30B-A3B/ --new_model_dir /mmu_mllm_hdd_2/lingzhixin/models/Keye-30B-A3B-scratch_0612 --delete_layer_mlp 1
  main()




"""
python3 /llm_reco/lingzhixin/recovlm_qw0510/recovlm/tools/model_helpers/hf_converters/convert_qwen3_to_keye.py --delete_mlp 0 --model_dir /llm_reco_ssd/zhouyang12/models/Keye-8B-demo_hf_vit_rope_slowfast_0608/ --vit_model_path /llm_reco_ssd/zhouyang12/models/Keye-8B-demo_hf_vit_rope_slowfast_0608/ --new_model_dir /llm_reco_ssd/zhouyang12/models/Keye-8B-demo_hf_vit_rope_slowfast_0608_split
"""
