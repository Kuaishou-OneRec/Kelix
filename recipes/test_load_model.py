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

# Set device - use CUDA if available, otherwise CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

processor = Qwen2_5_VLProcessor_moonvit.from_pretrained(
  "/llm_reco_ssd/zhouyang12/models/Qwen2-VL-7B-Instruct"
)


image = torch.randint(0, 255, (224, 224, 3), dtype=torch.uint8)
image = Image.fromarray(image.numpy())
images =[image]
texts = ["hello world"]
data = processor(images=images, text=texts)
print(data.keys())

# Create a properly formatted image_grid_thw tensor
# It should contain time, height, width dimensions for each image
# For a single image without time dimension, we use [1, H/patch_size, W/patch_size]
input_ids = data["input_ids"]
pixel_values = data["pixel_values"]
image_grid_thw = data["image_grid_thw"]
print(image_grid_thw)
# Correctly format image_grid_thw - assuming patch size is 16 for both H and W

# Convert inputs to tensors if they're not already
if not isinstance(input_ids, torch.Tensor):
    input_ids = torch.tensor(input_ids)
if not isinstance(pixel_values, torch.Tensor):
    pixel_values = torch.tensor(pixel_values)
if not isinstance(image_grid_thw, torch.Tensor):
    image_grid_thw = torch.tensor(image_grid_thw)

# Move tensors to the same device as the model
input_ids = input_ids.to(device)
pixel_values = pixel_values.to(device)
image_grid_thw = image_grid_thw.to(device)

rets = model(input_ids=input_ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw)
print(rets)
