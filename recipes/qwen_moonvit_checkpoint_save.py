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


def save_model_state(dict_state):
    torch.save(dict_state, "/llm_reco/maosiyang/model/qwen_moonvit/qwen2_5_vl_moonvit_state_dict.pth")

# 加载模型状态的示例代码
def load_model_state():
    # 1. 首先初始化一个模型实例
    loaded_model = Qwen2_5_VLForConditionalGeneration_moonvit.from_pretrained(
        "/llm_reco_ssd/zhouyang12/models/Qwen2-VL-7B-Instruct",
        ignore_mismatched_sizes=True
    )
    # 2. 加载保存的state dict
    state_dict = torch.load("/llm_reco/maosiyang/model/qwen_moonvit/qwen2_5_vl_moonvit_state_dict.pth")
    
    # 3. 将state dict加载到模型中
    print('=================================')
    loaded_model.load_state_dict(state_dict)
    for key, value in loaded_model.named_parameters():
        print('--------------------------------')
        print(key, value.shape) 
        print(value)
        print('--------------------------------')
    print('=================================')   
    
    return loaded_model 


if __name__ == "__main__":

    model = Qwen2_5_VLForConditionalGeneration_moonvit.from_pretrained(
      "/llm_reco_ssd/zhouyang12/models/Qwen2-VL-7B-Instruct",ignore_mismatched_sizes=True
    )


    
    pt = '/llm_reco/liuyang76/Models/MoonVitParam/MoonVit.pt'
    visual_state_dict = torch.load(pt)
    for key, value in visual_state_dict.items():
        print('--------------------------------')
        print(key, value.shape)
        print(value)
        print('--------------------------------')
    model.visual.load_state_dict(visual_state_dict)
    dict_state = model.state_dict()
    save_model_state(dict_state)
    loaded_model = load_model_state()
    for key, value in model.named_parameters():
        print(key, value.shape)
    print("--------------------------------")
    for key, value in loaded_model.named_parameters():
        print(key, value.shape)
