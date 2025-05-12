from recovlm.models.qwen_2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration_moonvit,Qwen2_5_VLForConditionalGeneration_siglip
from recovlm.models.qwen_2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor_moonvit,Qwen2_5_VLProcessor_siglip
import torch
from PIL import Image
from recipes.ViT.training.models.MoonVision.image_processing_kimi_vl import KimiVLImageProcessor_for_qwen2_5_vl
from qwen_vl_utils import process_vision_info
from recipes.ViT.training.models.MoonVision.modeling_kimi_vl import MoonVitPretrainedModel
from recovlm.models.qwen_3_vl.modeling_qwen3_vl import Qwen3_VLForConditionalGeneration_siglip

def save_model_state(dict_state):
    torch.save(dict_state, "/llm_reco/maosiyang/model/qwen_moonvit/qwen3_vl_siglip_state_dict.pth")

# 加载模型状态的示例代码
def load_model_state():
    # 1. 首先初始化一个模型实例
    loaded_model = Qwen2_5_VLForConditionalGeneration_siglip.from_pretrained(
        "/llm_reco_ssd/zhouyang12/models/Qwen3-8B-Base",
        ignore_mismatched_sizes=True
    )
    # 2. 加载保存的state dict
    state_dict = torch.load("/llm_reco/maosiyang/model/qwen_moonvit/qwen3_vl_siglip_state_dict.pth")
    
    # 3. 将state dict加载到模型中
    print('=================================')
    loaded_model.load_state_dict(state_dict)
    for key, value in loaded_model.named_parameters():
        print(key, value.shape) 
    print('=================================')   
    
    return loaded_model 


if __name__ == "__main__":

    model = Qwen2_5_VLForConditionalGeneration_siglip.from_pretrained(
      "/llm_reco_ssd/zhouyang12/models/Qwen3-8B-Base",ignore_mismatched_sizes=True
    )
    from safetensors import safe_open

    with safe_open("/llm_reco/liuyang76/Models/siglip2-so400m-patch14-384/model.safetensors", framework="pt", device="cpu") as f:
        pt = {}
        for key in f.keys():
            if "packing" in key:
                continue
            pt[key] = f.get_tensor(key)
    visual_state_dict = pt
    # Create a list of keys to iterate over
    keys_to_remove = []
    for key in visual_state_dict.keys():
        if "text_model" in key or "logit_scale" or "logit_bias"in key:
            keys_to_remove.append(key)
    
    # Remove the keys after iteration
    for key in keys_to_remove:
        del visual_state_dict[key]
        
    for key, value in visual_state_dict.items():
        print(key, value.shape)

    model.visual.load_state_dict(visual_state_dict,strict=False)
    dict_state = model.state_dict()
    save_model_state(dict_state)



    loaded_model = load_model_state()
    # Check if the visual parameters in loaded_model match those in visual_state_dict
    matched_count = 0
    mismatched_count = 0
    model_state_dict = loaded_model.state_dict()
    for key, value in loaded_model.named_parameters():
        if 'visual' in key:
            if key not in visual_state_dict:
                key = key.replace('visual.', '')
                if key not in visual_state_dict:
                    print(f"Warning: Key {key} not found in visual_state_dict")
                    continue
                
            is_equal = torch.allclose(value, visual_state_dict[key], rtol=1e-5, atol=1e-5)
            if is_equal:
                matched_count += 1
                # print(f"✓ {key}: Parameters match")
            else:
                mismatched_count += 1
                # print(f"✗ {key}: Parameters differ")
                # Calculate and print the difference statistics
                diff = torch.abs(value - visual_state_dict[key])
                # print(f"  Max difference: {diff.max().item():.6f}")
                # print(f"  Mean difference: {diff.mean().item():.6f}")


    print(f"\nSummary: {matched_count} parameters match, {mismatched_count} parameters differ")
    
    # Compare parameters between original model and saved state dict
    print("\nComparing original model with saved state dict:")
    original_model = Qwen2_5_VLForConditionalGeneration_siglip.from_pretrained(
        "/llm_reco_ssd/zhouyang12/models/Qwen3-8B-Base",
        ignore_mismatched_sizes=True
    )
    saved_state_dict = torch.load("/llm_reco/maosiyang/model/qwen_moonvit/qwen3_vl_siglip_state_dict.pth")
    
    orig_matched = 0
    orig_mismatched = 0
    for key, value in original_model.named_parameters():
        if key in saved_state_dict:
            is_equal = torch.allclose(value, saved_state_dict[key], rtol=1e-5, atol=1e-5)
            if is_equal:
                orig_matched += 1
                print(f"✓ {key}: Parameters match")
            else:
                orig_mismatched += 1
                print(f"✗ {key}: Parameters differ")
                diff = torch.abs(value - saved_state_dict[key])
                print(f"  Max difference: {diff.max().item():.6f}")
                print(f"  Mean difference: {diff.mean().item():.6f}")
        else:
            print(f"Warning: Key {key} not found in saved state dict")
    
    print(f"\nOriginal Model Comparison Summary: {orig_matched} parameters match, {orig_mismatched} parameters differ")
    