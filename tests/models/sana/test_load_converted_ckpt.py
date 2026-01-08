# /mmu_mllm_hdd_2/zangdunju/output2/RecoVLM/DiTSFT/batch6_324_1024_more_data/global_step80000/muse_converted/model.safetensors


from muse.models.base import Model
import torch


device=0
dtype=torch.float16
model_dir = "/mmu_mllm_hdd_2/zangdunju/output2/RecoVLM/DiTSFT/batch6_324_1024_more_data/global_step80000/muse_converted/"

model = Model.from_pretrained(model_dir)
model = model.to(device=device, dtype=dtype).eval()
print(f"  Model loaded: {type(model).__name__}")





