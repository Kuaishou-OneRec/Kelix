from recovlm.models.qwen_2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration_moonvit,Qwen2_5_VLForConditionalGeneration
from recovlm.models.qwen_2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor_moonvit 
import torch
from PIL import Image
from recipes.ViT.training.models.MoonVision.image_processing_kimi_vl import KimiVLImageProcessor_for_qwen2_5_vl
from qwen_vl_utils import process_vision_info
from recipes.ViT.training.models.MoonVision.modeling_kimi_vl import MoonVitPretrainedModel
# # from recovlm.models.qwen_2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration_moonvit
import json


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

model2 = Qwen2_5_VLForConditionalGeneration.from_pretrained(
  "/llm_reco_ssd/zhouyang12/models/Qwen2-VL-7B-Instruct",ignore_mismatched_sizes=True
)
model2.eval()


device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device,dtype=torch.bfloat16)

processor = Qwen2_5_VLProcessor_moonvit.from_pretrained(
  "/llm_reco_ssd/zhouyang12/models/Qwen2-VL-7B-Instruct"
)

print('--------------------------------')
for name, param in model.visual.named_parameters():
    print(name, param.shape)
print('--------------------------------')

for name,param in model2.visual.named_parameters():
    print(name, param.shape)
print('--------------------------------') 






# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "image",
#                 "image": "/llm_reco/penghao03/tmp/other/demo.jpeg",
#             },
#             {"type": "text", "text": "Describe this image."},
#         ],
#     }
# ]

# # Preparation for inference
# text = processor.apply_chat_template(
#     messages, tokenize=False, add_generation_prompt=True
# )
# image_inputs, video_inputs = process_vision_info(messages)



# # data2 = processor2(images,return_tensors="pt")
# data = processor(
#     text=[text],
#     images=image_inputs,
#     videos=video_inputs,
#     padding=True,
#     return_tensors="pt",
# )
# print(data.keys())


# # Create a properly formatted image_grid_thw tensor
# # It should contain time, height, width dimensions for each image
# # For a single image without time dimension, we use [1, H/patch_size, W/patch_size]
# input_ids = data["input_ids"]
# print('input_ids',input_ids)
# pixel_values = data["pixel_values"]
# image_grid_thw = data["image_grid_thw"]
# print(image_grid_thw)
# # Correctly format image_grid_thw - assuming patch size is 16 for both H and W

# # Convert inputs to tensors if they're not already
# if not isinstance(input_ids, torch.Tensor):
#     input_ids = torch.tensor(input_ids)
# if not isinstance(pixel_values, torch.Tensor):
#     pixel_values = torch.tensor(pixel_values)
# if not isinstance(image_grid_thw, torch.Tensor):
#     image_grid_thw = torch.tensor(image_grid_thw)

# # Move tensors to the same device as the model
# input_ids = input_ids.to(device)
# pixel_values = pixel_values.to(device)
# image_grid_thw = image_grid_thw.to(device)
# # image_grid_thw = data2.image_grid_thw.to(device)
# # pixel_values = data2.pixel_values.to(device)

# rets = model(input_ids=input_ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw)
# print(rets)
