saved_state_dict1 = torch.load("/llm_reco/maosiyang/model/qwen_moonvit/qwen3_vl_siglip_state_dict.pth")


file_path = "/llm_reco_ssd/zhouyang12/models/msy_Qwen3vl-8B-Base/model-00001-of-00005.safetensors"
saved_state_dict2 = safetensors.torch.load_file(file_path)

# 获取两个state dict的共同key
common_keys = set(saved_state_dict1.keys()) & set(saved_state_dict2.keys())
print(f"共同key的数量: {len(common_keys)}")

# 比较共同key的参数是否一致
diff_keys = []
for key in common_keys:
    if not torch.allclose(saved_state_dict1[key], saved_state_dict2[key], rtol=1e-5, atol=1e-5):
        diff_keys.append(key)

if diff_keys:
    print("\n参数不一致的key:")
    for key in diff_keys:
        print(f"- {key}")
else:
    print("\n所有共同key的参数都一致")
