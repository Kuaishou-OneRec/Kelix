import torch
import torch.nn as nn
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# 导入两个模型类
from muse.models.keye_ar.token_decoder_ori import PureDecoderTransformer
from muse.models.keye_ar.unified_token_decoder import TokenDecoder


def test_convert_hf_state_dict():
    """
    测试convert_hf_state_dict函数的正确性
    """
    print("开始测试convert_hf_state_dict函数...")
    
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
        "attention_function": "eager"  # 新模型使用eager attention，对应原始模型的use_flash_attn=False
    }
    
    # 2. 初始化原始模型和新模型
    print("\n初始化模型...")
    
    # 创建共享的token embedding
    token_embedding = nn.Embedding(config["vocab_size"], config["d_model"])
    
    # 初始化原始模型 (PureDecoderTransformer)
    ori_model = PureDecoderTransformer(
        vocab_size=config["vocab_size"],
        max_length=config["max_length"],
        d_model=config["d_model"],
        eos_token=config["eos_token"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        dim_feedforward=config["dim_feedforward"],
        token_embedding=token_embedding,
        use_flash_attn=False,  # 使用eager attention以匹配新模型
        use_gradient_checkpointing=config["use_gradient_checkpointing"],
        reduce=config["reduce"],
        lm_head=None  # 不使用lm_head以简化测试
    )
    
    # 初始化新模型 (TokenDecoder)
    new_model = TokenDecoder(
        vocab_size=config["vocab_size"],
        max_length=config["max_length"],
        d_model=config["d_model"],
        eos_token=config["eos_token"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        dim_feedforward=config["dim_feedforward"],
        token_embedding=token_embedding,
        use_gradient_checkpointing=config["use_gradient_checkpointing"],
        reduce=config["reduce"],
        attention_function=config["attention_function"]
    )
    
    # 3. 转换状态字典
    print("\n转换状态字典...")
    ori_state_dict = ori_model.state_dict()
    print(f"原始模型状态字典键数: {len(ori_state_dict)}")
    
    # 使用新模型的convert_hf_state_dict方法转换状态字典
    converted_state_dict = TokenDecoder.convert_hf_state_dict(ori_state_dict)
    print(f"转换后模型状态字典键数: {len(converted_state_dict)}")
    
    # 4. 将转换后的状态字典加载到新模型中
    print("\n加载转换后的状态字典到新模型...")
    new_model.load_state_dict(converted_state_dict)
    
    # 5. 比较两个模型的前向结果
    print("\n比较前向结果...")
    
    # 创建测试输入
    batch_size = 2
    seq_len = 5
    input_embeddings = torch.randn(batch_size, seq_len, config["d_model"])
    
    # 设置为评估模式
    ori_model.eval()
    new_model.eval()
    
    # 禁用梯度计算
    with torch.no_grad():
        # 获取原始模型的输出
        ori_output = ori_model(input_embeddings)
        print(f"原始模型输出形状: {ori_output.shape}")
        
        # 获取新模型的输出
        new_output = new_model(input_embeddings)
        print(f"新模型输出形状: {new_output.shape}")
        
        # 计算输出差异
        diff = torch.abs(ori_output - new_output)
        max_diff = torch.max(diff)
        mean_diff = torch.mean(diff)
        
        print(f"输出最大差异: {max_diff.item()}")
        print(f"输出平均差异: {mean_diff.item()}")
        
        # 检查差异是否在可接受范围内
        if max_diff < 1e-5 and mean_diff < 1e-6:
            print("\n✅ 测试通过！两个模型的前向结果相同。")
            return True
        else:
            print("\n❌ 测试失败！两个模型的前向结果存在差异。")
            
            # 打印差异较大的位置
            large_diff_indices = torch.where(diff > 1e-5)
            if len(large_diff_indices[0]) > 0:
                print(f"差异较大的位置数量: {len(large_diff_indices[0])}")
                # 打印前5个差异较大的位置
                for i in range(min(5, len(large_diff_indices[0]))):
                    batch_idx = large_diff_indices[0][i].item()
                    seq_idx = large_diff_indices[1][i].item()
                    dim_idx = large_diff_indices[2][i].item()
                    print(f"位置 ({batch_idx}, {seq_idx}, {dim_idx}): 原始值 = {ori_output[batch_idx, seq_idx, dim_idx].item()}, "
                          f"新值 = {new_output[batch_idx, seq_idx, dim_idx].item()}, 差异 = {diff[batch_idx, seq_idx, dim_idx].item()}")
            
            return False


if __name__ == "__main__":
    success = test_convert_hf_state_dict()
    sys.exit(0 if success else 1)