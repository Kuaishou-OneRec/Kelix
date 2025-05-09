from recovlm.models.qwen_2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration_moonvit,Qwen2_5_VLForConditionalGeneration_siglip
from recovlm.models.qwen_2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor_moonvit,Qwen2_5_VLProcessor_siglip
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
    torch.save(dict_state, "/llm_reco/maosiyang/model/qwen_moonvit/qwen2_5_vl_siglip_state_dict.pth")

# 加载模型状态的示例代码
def load_model_state():
    # 1. 首先初始化一个模型实例
    loaded_model = Qwen2_5_VLForConditionalGeneration_siglip.from_pretrained(
        "/llm_reco_ssd/zhouyang12/models/Qwen2.5-VL-7B-Instruct",
        ignore_mismatched_sizes=True
    )
    # 2. 加载保存的state dict
    state_dict = torch.load("/llm_reco/maosiyang/model/qwen_moonvit/qwen2_5_vl_siglip_state_dict.pth")
    
    # 3. 将state dict加载到模型中
    print('=================================')
    loaded_model.load_state_dict(state_dict)
    for key, value in loaded_model.named_parameters():
        print(key, value.shape) 
    print('=================================')   
    
    return loaded_model 


if __name__ == "__main__":

    model = Qwen2_5_VLForConditionalGeneration_siglip.from_pretrained(
      "/llm_reco_ssd/zhouyang12/models/Qwen2.5-VL-7B-Instruct",ignore_mismatched_sizes=True
    )
    from safetensors import safe_open

    with safe_open("/llm_reco/liuyang76/Models/siglip2-so400m-patch14-384/model.safetensors", framework="pt", device="cpu") as f:
        pt = {}
        for key in f.keys():
            if "packing" in key:
                continue
            pt[key] = f.get_tensor(key)
    # ckpt = '/llm_reco/liuyang76/Models/siglip2-so400m-patch14-384/model.safetensors'
    # #convert to pt
    # pt = '/llm_reco/liuyang76/Models/MoonVitParam/MoonVit.pt'
    visual_state_dict = pt
    # for key, value in visual_state_dict.items():
    #     print('--------------------------------')
    #     print(key, value.shape)
    #     print(value)
    #     print('--------------------------------')
    model.visual.load_state_dict(visual_state_dict,strict=False)
    dict_state = model.state_dict()
    save_model_state(dict_state)
    loaded_model = load_model_state()
    # Check if the visual parameters in loaded_model match those in visual_state_dict
    matched_count = 0
    mismatched_count = 0
    for key, value in loaded_model.named_parameters():
        if 'visual' in key:
            if key not in visual_state_dict:
                #Warning: Key visual.encoder.blocks.26.wo.bias not found in visual_state_dict
                #I want to delete visual. from key
                key = key.replace('visual.', '')
                if key not in visual_state_dict:
                    print(f"Warning: Key {key} not found in visual_state_dict")
                    continue
                
            is_equal = torch.allclose(value, visual_state_dict[key], rtol=1e-5, atol=1e-5)
            if is_equal:
                matched_count += 1
                print(f"✓ {key}: Parameters match")
            else:
                mismatched_count += 1
                print(f"✗ {key}: Parameters differ")
                # Calculate and print the difference statistics
                diff = torch.abs(value - visual_state_dict[key])
                print(f"  Max difference: {diff.max().item():.6f}")
                print(f"  Mean difference: {diff.mean().item():.6f}")
    
    print(f"\nSummary: {matched_count} parameters match, {mismatched_count} parameters differ")