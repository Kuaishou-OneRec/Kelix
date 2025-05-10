import argparse
import torch
from safetensors import safe_open

siglip_path = "/llm_reco/liuyang76/Models/siglip2-so400m-patch14-384/model.safetensors"

with safe_open(siglip_path, framework="pt", device="cpu") as f:
    tensors = {}
    for key in f.keys():
        if key.lower().startswith("moonvit"):
            continue
        if "logit" in key.lower():
            continue
        if "text_model" in key.lower():
            continue
        tensors["visual." + key] = f.get_tensor(key)

tensors["visual.vision_model.embeddings.packing_position_embedding.weight"] = torch.zeros((32768, 1152), dtype=torch.float32)

model = torch.load("/llm_reco/maosiyang/model/qwen_moonvit/qwen2_5_vl_moonvit_state_dict.pth", map_location="cpu")
for key in model:
    if key.split(".")[0].lower() != "visual":
        tensors[key] = model[key]
from collections import Counter
print(Counter([x.split(".")[0] for x in tensors]))
print(Counter(tensors[x].dtype for x in tensors))
torch.save(tensors, "/llm_reco_ssd/zangdunju/output2/RecoVLM/SigLIP/siglip/global_step1000/model_float32.pth")
