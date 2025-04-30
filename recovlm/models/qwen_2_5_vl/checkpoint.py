from typing import Dict, Any
import tqdm
import re
import torch
from recovlm.training.checkpoint import CheckpointConverter
from recovlm.models.qwen_2_5_vl.configuration_qwen_2_5_vl import Qwen2_5_VLVisionConfig
from recipes.ViT.training.models.MoonVision.configuration_kimi_vl import MoonViTConfig

class Qwen2VLCheckpointConverter(CheckpointConverter):
  def __init__(self, model_path_or_name: str):
    self.model_path_or_name = model_path_or_name
    self.config = Qwen2_5_VLVisionConfig.from_pretrained(model_path_or_name)

  def __call__(self,
               state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    num_heads = self.config.vision_config.num_heads
    hidden_size = self.config.vision_config.embed_dim
    print(f"Converting from {self.model_path_or_name}")
    for k, v in tqdm.tqdm(state_dict.items()):
      if re.match(r"visual\.blocks\.\d+\.attn\.qkv\.weight", k):
        state_dict[k] = v.reshape(
          3, num_heads, hidden_size // num_heads, hidden_size
        ).permute(1, 0, 2, 3).reshape(hidden_size * 3, hidden_size)
        print(f"Convert: {k}")
      elif re.match(r"visual\.blocks\.\d+\.attn\.qkv\.bias", k):
        state_dict[k] = v.reshape(
          3, num_heads, hidden_size // num_heads
        ).permute(1, 0, 2).reshape(hidden_size * 3)
        print(f"Convert: {k}")
    return state_dict


  def convert(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
      return self.__call__(state_dict)
  
  def revert(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
      return self.tp_to_original(state_dict)

  def tp_to_original(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
      num_heads = self.config.vision_config.num_heads
      hidden_size = self.config.vision_config.embed_dim
      print(f"Reverting weights to original format for {self.model_path_or_name}")
      
      for k, v in tqdm.tqdm(state_dict.items()):
          if re.match(r"visual\.blocks\.\d+\.attn\.qkv\.weight", k):
              # 逆向操作：将[heads*3*head_dim, hidden_size]转回[3*hidden_size, hidden_size]
              state_dict[k] = v.reshape(
                  num_heads, 3, hidden_size // num_heads, hidden_size
              ).permute(1, 0, 2, 3).reshape(3 * hidden_size, hidden_size)
              print(f"Reverted: {k}")
              
          elif re.match(r"visual\.blocks\.\d+\.attn\.qkv\.bias", k):
              # 逆向操作：将[heads*3*head_dim]转回[3*hidden_size]
              state_dict[k] = v.reshape(
                  num_heads, 3, hidden_size // num_heads
              ).permute(1, 0, 2).reshape(3 * hidden_size)
              print(f"Reverted: {k}")
      
      return state_dict



def _test_convert():
    from recovlm.training.checkpoint import load_hf_checkpoint
    model_dir = "/llm_reco_ssd/zhouyang12/models/Qwen2-VL-7B-Instruct/"
    state_dict = load_hf_checkpoint(model_dir)
    converter = Qwen2VLCheckpointConverter(model_dir)
    
    # 1. 测试单个权重张量的转换可逆性
    print("\n=== 测试单个张量转换的可逆性 ===")
    test_key = None
    for k in state_dict.keys():
        if re.match(r"visual\.blocks\.0\.attn\.qkv\.weight", k):
            test_key = k
            break
    
    if test_key:
        original_tensor = state_dict[test_key].clone()
        print(f"测试张量: {test_key}, 原始形状: {original_tensor.shape}")
        
        # 正向转换
        converted_tensor = converter({test_key: original_tensor.clone()})[test_key]
        print(f"转换后形状: {converted_tensor.shape}")
        
        # 逆向转换
        reverted_tensor = converter.tp_to_original({test_key: converted_tensor.clone()})[test_key]
        print(f"还原后形状: {reverted_tensor.shape}")
        
        # 验证数值一致性
        assert torch.allclose(original_tensor, reverted_tensor, atol=1e-6), "转换不可逆！"
        print("✅ 单个张量转换测试通过")
    
    # 2. 测试完整state_dict的转换可逆性
    print("\n=== 测试完整state_dict转换的可逆性 ===")
    original_dict = {k: v.clone() for k, v in state_dict.items()}
    
    # 正向转换
    converted_dict = converter(original_dict.copy())
    
    # 逆向转换
    reverted_dict = converter.tp_to_original(converted_dict.copy())
    
    # 验证所有键的一致性
    assert set(original_dict.keys()) == set(reverted_dict.keys()), "键不匹配！"
    
    # 验证所有张量的数值一致性
    mismatch_count = 0
    for k in original_dict:
        if not torch.allclose(original_dict[k], reverted_dict[k], atol=1e-6):
            print(f"数值不匹配: {k}")
            mismatch_count += 1
    
    if mismatch_count == 0:
        print("✅ 完整state_dict转换测试通过")
    else:
        print(f"⚠️ 发现{mismatch_count}个张量数值不匹配")
    
    # 3. 验证转换后的权重是否可用于模型初始化
    print("\n=== 验证转换后权重的可用性 ===")
    from recovlm.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_dir)
    
    try:
        model.load_state_dict(converted_dict, strict=False)
        print("✅ 转换后权重加载测试通过 (strict=False)")
        
        # 严格模式加载测试
        model.load_state_dict(converted_dict, strict=True)
        print("✅ 转换后权重加载测试通过 (strict=True)")
    except Exception as e:
        print(f"❌ 权重加载失败: {str(e)}")
    
    print("\n=== 测试完成 ===")



##TODO:将config和block名称转换成moonvit的

class Qwen2_5_VL_moonvitCheckpointConverter(CheckpointConverter):
  def __init__(self, model_path_or_name: str):
    self.model_path_or_name = model_path_or_name
    self.config = Qwen2_5_VLVisionConfig.from_pretrained(model_path_or_name)

  def __call__(self,
               state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    num_heads = self.config.vision_config.num_heads
    hidden_size = self.config.vision_config.embed_dim
    print(f"Converting from {self.model_path_or_name}")
    for k, v in tqdm.tqdm(state_dict.items()):
      if re.match(r"visual\.encoder\.blocks\.\d+\.wqkv\.weight", k):
        state_dict[k] = v.reshape(
          3, num_heads, hidden_size // num_heads, hidden_size
        ).permute(1, 0, 2, 3).reshape(hidden_size * 3, hidden_size)
        print(f"Convert: {k}")
      elif re.match(r"visual\.encoder\.blocks\.\d+\.wqkv\.bias", k):
        state_dict[k] = v.reshape(
          3, num_heads, hidden_size // num_heads
        ).permute(1, 0, 2).reshape(hidden_size * 3)
        print(f"Convert: {k}")
    return state_dict


  def convert(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
      return self.__call__(state_dict)
  
  def revert(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
      return self.tp_to_original(state_dict)

  def tp_to_original(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
      num_heads = self.config.vision_config.num_heads
      hidden_size = self.config.vision_config.embed_dim
      print(f"Reverting weights to original format for {self.model_path_or_name}")
      
      for k, v in tqdm.tqdm(state_dict.items()):
          if re.match(r"visual\.encoder\.blocks\.\d+\.wqkv\.weight", k):
              # 逆向操作：将[heads*3*head_dim, hidden_size]转回[3*hidden_size, hidden_size]
              state_dict[k] = v.reshape(
                  num_heads, 3, hidden_size // num_heads, hidden_size
              ).permute(1, 0, 2, 3).reshape(3 * hidden_size, hidden_size)
              print(f"Reverted: {k}")
              
          elif re.match(r"visual\.encoder\.blocks\.\d+\.wqkv\.bias", k):
              # 逆向操作：将[heads*3*head_dim]转回[3*hidden_size]
              state_dict[k] = v.reshape(
                  num_heads, 3, hidden_size // num_heads
              ).permute(1, 0, 2).reshape(3 * hidden_size)
              print(f"Reverted: {k}")
      
      return state_dict



def _test_convert():
    from recovlm.training.checkpoint import load_hf_checkpoint
    model_dir = "/llm_reco_ssd/zhouyang12/models/Qwen2-VL-7B-Instruct/"
    model = Qwen2_5_VLForConditionalGeneration_moonvit.from_pretrained(
    "/llm_reco_ssd/zhouyang12/models/Qwen2-VL-7B-Instruct",ignore_mismatched_sizes=True
    )
    state_dict = model.state_dict()
    converter = Qwen2_5_VL_moonvitCheckpointConverter(model_dir)
    
    # 1. 测试单个权重张量的转换可逆性
    print("\n=== 测试单个张量转换的可逆性 ===")
    test_key = None
    for k in state_dict.keys():
        if re.match(r"visual\.encoder\.blocks\.0\.wqkv\.weight", k):
            test_key = k
            break
    
    if test_key:
        original_tensor = state_dict[test_key].clone()
        print(f"测试张量: {test_key}, 原始形状: {original_tensor.shape}")
        
        # 正向转换
        converted_tensor = converter({test_key: original_tensor.clone()})[test_key]
        print(f"转换后形状: {converted_tensor.shape}")
        
        # 逆向转换
        reverted_tensor = converter.tp_to_original({test_key: converted_tensor.clone()})[test_key]
        print(f"还原后形状: {reverted_tensor.shape}")
        
        # 验证数值一致性
        assert torch.allclose(original_tensor, reverted_tensor, atol=1e-6), "转换不可逆！"
        print("✅ 单个张量转换测试通过")
    
    # 2. 测试完整state_dict的转换可逆性
    print("\n=== 测试完整state_dict转换的可逆性 ===")
    original_dict = {k: v.clone() for k, v in state_dict.items()}
    
    # 正向转换
    converted_dict = converter(original_dict.copy())
    
    # 逆向转换
    reverted_dict = converter.tp_to_original(converted_dict.copy())
    
    # 验证所有键的一致性
    assert set(original_dict.keys()) == set(reverted_dict.keys()), "键不匹配！"
    
    # 验证所有张量的数值一致性
    mismatch_count = 0
    for k in original_dict:
        if not torch.allclose(original_dict[k], reverted_dict[k], atol=1e-6):
            print(f"数值不匹配: {k}")
            mismatch_count += 1
    
    if mismatch_count == 0:
        print("✅ 完整state_dict转换测试通过")
    else:
        print(f"⚠️ 发现{mismatch_count}个张量数值不匹配")
    
    # 3. 验证转换后的权重是否可用于模型初始化
    print("\n=== 验证转换后权重的可用性 ===")
    from recovlm.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration_moonvit
    model = Qwen2_5_VLForConditionalGeneration_moonvit.from_pretrained(model_dir)
    
    try:
        model.load_state_dict(converted_dict, strict=False)
        print("✅ 转换后权重加载测试通过 (strict=False)")
        
        # 严格模式加载测试
        model.load_state_dict(converted_dict, strict=True)
        print("✅ 转换后权重加载测试通过 (strict=True)")
    except Exception as e:
        print(f"❌ 权重加载失败: {str(e)}")
    
    print("\n=== 测试完成 ===")





if __name__ == "__main__":
    _test_convert()


