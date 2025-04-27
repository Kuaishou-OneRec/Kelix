from recovlm.models.qwen_2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration_moonvit
from recovlm.models.qwen_2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor_moonvit 
import torch
from PIL import Image


# from recovlm.models.qwen_2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration_moonvit
# import json


# #config = json.load(open("/llm_reco_ssd/zhouyang12/models/Qwen2.5-7B-Instruct/config.json", "r"))
# #model = Qwen2_5_VLForConditionalGeneration_moonvit(config)
# model = \
# Qwen2_5_VLForConditionalGeneration_moonvit.from_pretrained("/llm_reco_ssd/zhouyang12/models/Qwen2.5-7B-Instruct/",
#          ignore_mismatched_sizes=True)
# for key, value in model.named_parameters():
#     print(key, value.shape)








model = Qwen2_5_VLForConditionalGeneration_moonvit.from_pretrained(
  "/llm_reco_ssd/zhouyang12/models/Qwen2-VL-7B-Instruct",ignore_mismatched_sizes=True
)
model.eval()
processor = Qwen2_5_VLProcessor_moonvit.from_pretrained(
  "/llm_reco_ssd/zhouyang12/models/Qwen2-VL-7B-Instruct"
)


image = torch.randint(0, 255, (224, 224, 3), dtype=torch.uint8)
image = Image.fromarray(image.numpy())
images =[image]
texts = ["hello world"]
data = processor(images=images, text=texts)
rets = model(data)
print(rets)
