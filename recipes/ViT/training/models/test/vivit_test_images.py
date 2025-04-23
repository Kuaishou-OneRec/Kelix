from transformers import AutoModel, AutoImageProcessor
# 指定本地保存路径
local_path = "/llm_reco_ssd/zhouyang12/models/MoonViT-SO-400M"

# 下载并保存图像处理器和模型
model_path = "moonshotai/MoonViT-SO-400M"
model = AutoModel.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)
processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)

model.save_pretrained(local_path)
processor.save_pretrained(local_path)