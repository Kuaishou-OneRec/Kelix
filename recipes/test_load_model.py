# from recovlm.models.qwen_2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration_moonvit
# from recovlm.models.qwen_2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor_moonvit 
import torch
# from PIL import Image
from recipes.ViT.training.models.MoonVision.image_processing_kimi_vl import KimiVLImageProcessor_for_qwen2_5_vl

from recipes.ViT.training.models.MoonVision.modeling_kimi_vl import MoonVitPretrainedModel
# # from recovlm.models.qwen_2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration_moonvit
import json


# # #config = json.load(open("/llm_reco_ssd/zhouyang12/models/Qwen2.5-7B-Instruct/config.json", "r"))
# # #model = Qwen2_5_VLForConditionalGeneration_moonvit(config)
# # model = \
# # Qwen2_5_VLForConditionalGeneration_moonvit.from_pretrained("/llm_reco_ssd/zhouyang12/models/Qwen2.5-7B-Instruct/",
# #          ignore_mismatched_sizes=True)
# # for key, value in model.named_parameters():
# #     print(key, value.shape)

# model = Qwen2_5_VLForConditionalGeneration_moonvit.from_pretrained(
#   "/llm_reco_ssd/zhouyang12/models/Qwen2-VL-7B-Instruct",ignore_mismatched_sizes=True
# )
# model.eval()

# # Set device - use CUDA if available, otherwise CPU
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = model.to(device)

# processor = Qwen2_5_VLProcessor_moonvit.from_pretrained(
#   "/llm_reco_ssd/zhouyang12/models/Qwen2-VL-7B-Instruct"
# )

# processor2 = KimiVLImageProcessor_for_qwen2_5_vl()
# image = torch.randint(0, 255, (224, 224, 3), dtype=torch.uint8)
# image = Image.fromarray(image.numpy())
# images =[image]
# texts = ["hello world"]

# data2 = processor2(images,return_tensors="pt")
# data = processor(images=images, text=texts)
# print(data.keys())


# # Create a properly formatted image_grid_thw tensor
# # It should contain time, height, width dimensions for each image
# # For a single image without time dimension, we use [1, H/patch_size, W/patch_size]
# input_ids = data["input_ids"]
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
# image_grid_thw = data2.image_grid_thw.to(device)
# pixel_values = data2.pixel_values.to(device)

# rets = model(input_ids=input_ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw)
# print(rets)



from PIL import Image
from transformers import AutoModel, AutoImageProcessor
from recipes.ViT.training.models.MoonVision.configuration_kimi_vl import MoonViTConfig, KimiVLConfig
model_path = "moonshotai/MoonViT-SO-400M"
# 指定单一设备
device = "cuda:0" if torch.cuda.is_available() else "cpu"
MoonViT_config = MoonViTConfig()
MoonViT_config._attn_implementation = 'flash_attention_2'
model = MoonVitPretrainedModel(MoonViT_config).to(device,dtype=torch.bfloat16)
# model = AutoModel.from_pretrained(
#     model_path,
#     torch_dtype="auto",
#     device_map={"": device},  # 强制使用同一设备
#     trust_remote_code=True,
# )
processor = KimiVLImageProcessor_for_qwen2_5_vl.from_pretrained(model_path, trust_remote_code=True)

image = torch.randint(0, 255, (224, 224, 3), dtype=torch.uint8)
image = Image.fromarray(image.numpy())
images = [image]

data = processor(images, return_tensors="pt").to(device=model.device, dtype=model.dtype)
image_grid_hws = [(data.image_grid_thw[0][1],data.image_grid_thw[0][2])]
image_grid_hws = torch.tensor(image_grid_hws, dtype=torch.int32, device=model.device)
image_features: list = model(data.pixel_values, image_grid_hws)

print(f"allalaladtype: {image_features[0].dtype}, shape: {image_features[0].shape}")
# dtype: torch.bfloat16, shape: torch.Size([1092, 4, 1152])
