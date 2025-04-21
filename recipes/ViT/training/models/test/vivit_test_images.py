from transformers import VivitImageProcessor, VivitModel

# 指定本地保存路径
local_path = "/llm_reco_ssd/zhouyang12/models/vivit-b-16x2-kinetics400"

# 下载并保存图像处理器和模型
image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
image_processor.save_pretrained(local_path)

model = VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400")
model.save_pretrained(local_path)