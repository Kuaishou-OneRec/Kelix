import torch
from safetensors.torch import save_file

# 加载原始 PyTorch 模型
model_path = "/mmu_mllm_hdd_2/zhouyang12/output2/Keye/0.8.0/ViT/80m/0.0.1/global_step19800/vision_model.pth"
state_dict = torch.load(model_path, map_location="cpu")  # 避免GPU依赖

# 保存为 SafeTensor 格式
output_path = "/mmu_mllm_hdd_2/zhouyang12/output2/Keye/0.8.0/ViT/80m/0.0.1/global_step19800/vision_model.safetensors"
save_file(state_dict, output_path)

print(f"模型已转换为 {output_path}")