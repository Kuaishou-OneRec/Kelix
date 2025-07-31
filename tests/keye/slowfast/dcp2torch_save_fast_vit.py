import json
import os
import os.path as osp
import argparse
from checkpoint import dcp_to_torch_save
import shutil
import time
import socket
import random
import numpy as np
import re
import torch
def convert_dir(checkpoint_dir, model_name):
    return osp.join(checkpoint_dir, model_name, f"global_{model_name}")


# /llm_reco_ssd/zhouyang12/models/Qwen3-1.7B-siglip

checkpoint_dir = "/mmu_mllm_hdd_2/zhouyang12/output1/Keye/0.9.1/Stage1/8b/0.0.1"
model_paths = ["step39000"]
model_paths = ['/mmu_mllm_hdd_2/zhouyang12/output2/Keye/0.8.0/ViT/80m/0.0.1/global_step19800']
for model_path in model_paths:
    real_model_path = convert_dir(checkpoint_dir, model_path)
    if not os.path.exists(real_model_path):
        os.makedirs(real_model_path)

    dcp_to_torch_save(real_model_path, f"{real_model_path}/hf", model_only=True,
                                        use_safetensor=True, max_gb_per_shard=10)

    #base_model_path = "/llm_reco_ssd/zhouyang12/models/Qwen3-1.7B-siglip"
    base_model_path = "/llm_reco_ssd/zhouyang12/models/Keye-8B-demo_hf_vit_rope"
    #base model path  
    # 将模型的配置文件全部复制到hf文件夹中
    for file in os.listdir(base_model_path):
        if file == "model.safetensors.index.json":
            continue
        if file.endswith('.json') or file.endswith('.py') or file.endswith('.txt'):
            shutil.copy(osp.join(base_model_path, file), osp.join(real_model_path, 'hf', file))