from recovlm.models.qwen_2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration_moonvit,Qwen2_5_VLForConditionalGeneration
from recovlm.models.qwen_2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor_moonvit 
import torch
from PIL import Image
from recipes.ViT.training.models.MoonVision.image_processing_kimi_vl import KimiVLImageProcessor_for_qwen2_5_vl
from qwen_vl_utils import process_vision_info
from recipes.ViT.training.models.MoonVision.modeling_kimi_vl import MoonVitPretrainedModel
# # from recovlm.models.qwen_2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration_moonvit
import json

from recipes.ViT.training.models.siglip.modeling_siglip import SiglipVisionModel

# #config = json.load(open("/llm_reco_ssd/zhouyang12/models/Qwen2.5-7B-Instruct/config.json", "r"))
# #model = Qwen2_5_VLForConditionalGeneration_moonvit(config)
# model = \
# Qwen2_5_VLForConditionalGeneration_moonvit.from_pretrained("/llm_reco_ssd/zhouyang12/models/Qwen2.5-7B-Instruct/",
#          ignore_mismatched_sizes=True)
# for key, value in model.named_parameters():
#     print(key, value.shape)



if __name__ == "__main__":

    model = SiglipVisionModel.from_pretrained(
      "/llm_reco/liuyang76/Models/siglip2-so400m-patch14-384",ignore_mismatched_sizes=True
    )
    for key, value in model.named_parameters():
        print(key, value.shape)


