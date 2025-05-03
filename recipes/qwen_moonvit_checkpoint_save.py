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

#save model dict state as what we can load use from pretrained 
dict_state = model.state_dict()
torch.save(dict_state, "/llm_reco/maosiyang/model/qwen_moonvit/qwen2_5_vl_moonvit_state_dict.pth") 