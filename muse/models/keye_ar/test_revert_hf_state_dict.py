import torch
import torch.nn as nn
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# 导入两个模型类
from muse.models.keye_ar.token_decoder_ori import PureDecoderTransformer
from muse.models.keye_ar.token_decoder import TokenDecoder


def test_revert_hf_state_dict_symmetry():
    """
    测试revert_hf_state_dict函数是否与convert_hf_state_dict完全对称
    """
    print("开始测试revert_hf_state_dict对称性...")
    
    # 1. 设置相同的配置参数
    config = {
        "vocab_size": 1000,
        "max_length": 30,
        "d_model": 128,
        "eos_token": 999,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 512,
        "use_gradient_checkpointing": False,
        "reduce": False,
        "attention_function": "eager"
    }
    
    # 2. 初始化原始模型
    print("\n初始化原始模型...")
    token_embedding = nn.Embedding(config["vocab_size"], config["d_model"])
    
    ori_model = PureDecoderTransformer(
        vocab_size=config["vocab_size"],
        max_length=config["max_length"],
        d_model=config["d_model"],
        eos_token=config["eos_token"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        dim_feedforward=config["dim_feedforward"],
        token_embedding=token_embedding,
        use_flash_attn=False,
        use_gradient_checkpointing=config["use_gradient_checkpointing"],
        reduce=config["reduce"],
        lm_head=None
    )
    
    # 3. 获取原始模型状态字典
    print("\n获取原始模型状态字典...")
    ori_state_dict = ori_model.state_dict()
    print(f"原始模型状态字典键数: {len(ori_state_dict)}")
    
    # 4. 转换为新模型状态字典
    print("\n转换为新模型状态字典...")
    converted_state_dict = TokenDecoder.convert_hf_state_dict(ori_state_dict, reduce_mode=config["reduce"])
    print(f"转换后模型状态字典键数: {len(converted_state_dict)}")
    
    # 5. 还原回原始模型状态字典
    print("\n还原回原始模型状态字典...")
    reverted_state_dict = TokenDecoder.revert_hf_state_dict(converted_state_dict, reduce_mode=config["reduce"])
    print(f"还原后模型状态字典键数: {len(reverted_state_dict)}")
    
    # 6. 比较原始状态字典和还原后的状态字典
    print("\n比较原始状态字典和还原后的状态字典...")
    
    # 检查键的数量是否一致
    keys_match = set(ori_state_dict.keys()) == set(reverted_state_dict.keys())
    print(f"键集合是否一致: {'✅ 是' if keys_match else '❌ 否'}")
    
    if not keys_match:
        print(f"原始键: {set(ori_state_dict.keys())}")
        print(f"还原键: {set(reverted_state_dict.keys())}")
        missing_in_reverted = set(ori_state_dict.keys()) - set(reverted_state_dict.keys())
        extra_in_reverted = set(reverted_state_dict.keys()) - set(ori_state_dict.keys())
        if missing_in_reverted:
            print(f"还原字典中缺失的键: {missing_in_reverted}")
        if extra_in_reverted:
            print(f"还原字典中多余的键: {extra_in_reverted}")
    
    # 检查每个键对应的张量是否一致
    tensors_match = True
    max_diff = 0.0
    diff_details = []
    
    for key in ori_state_dict.keys():
        if key in reverted_state_dict:
            ori_tensor = ori_state_dict[key]
            rev_tensor = reverted_state_dict[key]
            
            # 检查形状是否一致
            if ori_tensor.shape != rev_tensor.shape:
                print(f"键 {key} 形状不一致: 原始 {ori_tensor.shape} vs 还原 {rev_tensor.shape}")
                tensors_match = False
                continue
            
            # 计算差异
            diff = torch.abs(ori_tensor - rev_tensor)
            max_diff_tensor = torch.max(diff).item()
            mean_diff_tensor = torch.mean(diff).item()
            
            if max_diff_tensor > max_diff:
                max_diff = max_diff_tensor
            
            # 检查差异是否在可接受范围内
            if max_diff_tensor > 1e-6 or mean_diff_tensor > 1e-7:
                tensors_match = False
                diff_details.append((key, max_diff_tensor, mean_diff_tensor))
        else:
            print(f"键 {key} 在还原字典中不存在")
            tensors_match = False
    
    print(f"张量形状和数值是否一致: {'✅ 是' if tensors_match else '❌ 否'}")
    if max_diff > 0:
        print(f"最大差异: {max_diff}")
    if diff_details:
        print(f"存在显著差异的键数量: {len(diff_details)}")
        # 显示前几个差异较大的键
        for key, max_diff_val, mean_diff_val in diff_details[:5]:
            print(f"  {key}: 最大差异={max_diff_val:.2e}, 平均差异={mean_diff_val:.2e}")
    
    # 7. 测试转换-还原-再转换的对称性
    print("\n=== 测试转换-还原-再转换的对称性 ===")
    
    # 再次转换还原后的状态字典
    double_converted_state_dict = TokenDecoder.convert_hf_state_dict(reverted_state_dict, reduce_mode=config["reduce"])
    
    # 比较第一次转换和第二次转换的结果
    conversion_symmetric = True
    conversion_max_diff = 0.0
    
    for key in converted_state_dict.keys():
        if key in double_converted_state_dict:
            first_conversion = converted_state_dict[key]
            second_conversion = double_converted_state_dict[key]
            
            # 检查形状是否一致
            if first_conversion.shape != second_conversion.shape:
                print(f"二次转换键 {key} 形状不一致")
                conversion_symmetric = False
                continue
            
            # 计算差异
            diff = torch.abs(first_conversion - second_conversion)
            max_diff_tensor = torch.max(diff).item()
            
            if max_diff_tensor > conversion_max_diff:
                conversion_max_diff = max_diff_tensor
            
            # 检查差异是否在可接受范围内
            if max_diff_tensor > 1e-6:
                conversion_symmetric = False
                print(f"二次转换键 {key} 存在显著差异: {max_diff_tensor}")
        else:
            print(f"二次转换字典中缺失键 {key}")
            conversion_symmetric = False
    
    print(f"转换-还原-再转换是否对称: {'✅ 是' if conversion_symmetric else '❌ 否'}")
    if conversion_max_diff > 0:
        print(f"二次转换最大差异: {conversion_max_diff}")
    
    # 总体结果
    overall_success = keys_match and tensors_match and conversion_symmetric
    print(f"\n总体对称性测试结果: {'✅ 全部通过' if overall_success else '❌ 存在失败'}")
    
    return overall_success


if __name__ == "__main__":
    success = test_revert_hf_state_dict_symmetry()
    sys.exit(0 if success else 1)