import torch
import torch.nn as nn
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# 导入两个模型类
from muse.models.keye_ar.token_decoder_ori import PureDecoderTransformer
from muse.models.keye_ar.token_decoder import TokenDecoder


def test_infer_id_embs_consistency():
    """
    测试给定和不给定infer_id_embs_fn时新旧模型的一致性
    """
    print("开始测试infer_id_embs_fn一致性...")
    
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
    
    # 2. 创建共享的token embedding和测试输入
    token_embedding = nn.Embedding(config["vocab_size"], config["d_model"])
    
    # 创建测试输入
    batch_size = 2
    seq_len = 5
    test_input_ids = torch.randint(0, config["vocab_size"], (batch_size, seq_len))
    
    print("\n=== 测试1: 不给定infer_id_embs_fn时的一致性 ===")
    
    # 初始化原始模型 (PureDecoderTransformer) 不使用infer_id_embs_fn
    ori_model_no_fn = PureDecoderTransformer(
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
        lm_head=None,
        infer_id_embs_fn=None
    )
    
    # 初始化新模型 (TokenDecoder) 不使用infer_id_embs_fn
    new_model_no_fn = TokenDecoder(
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
        lm_head=None,
        infer_id_embs_fn=None,
        attention_function=config["attention_function"]
    )
    
    # 设置为评估模式
    ori_model_no_fn.eval()
    new_model_no_fn.eval()
    
    # 禁用梯度计算
    with torch.no_grad():
        # 获取原始模型的输出
        ori_output_no_fn = ori_model_no_fn.forward_with_tokens(test_input_ids)
        print(f"原始模型(无infer_id_embs_fn)输出形状: {ori_output_no_fn.shape}")
        
        # 获取新模型的输出
        new_output_no_fn = new_model_no_fn.forward_with_tokens(test_input_ids)
        print(f"新模型(无infer_id_embs_fn)输出形状: {new_output_no_fn.shape}")
        
        # 计算输出差异
        diff_no_fn = torch.abs(ori_output_no_fn - new_output_no_fn)
        max_diff_no_fn = torch.max(diff_no_fn)
        mean_diff_no_fn = torch.mean(diff_no_fn)
        
        print(f"无infer_id_embs_fn时输出最大差异: {max_diff_no_fn.item()}")
        print(f"无infer_id_embs_fn时输出平均差异: {mean_diff_no_fn.item()}")
        
        # 检查差异是否在可接受范围内
        consistency_no_fn = max_diff_no_fn < 1e-5 and mean_diff_no_fn < 1e-6
        print(f"无infer_id_embs_fn时一致性测试: {'✅ 通过' if consistency_no_fn else '❌ 失败'}")
    
    print("\n=== 测试2: 给定infer_id_embs_fn时的一致性 ===")
    
    # 定义一个简单的infer_id_embs_fn示例
    def infer_id_embs_fn(ids):
        # 使用标准embedding，但在实际应用中可能有更复杂的逻辑
        return token_embedding(ids)
    
    # 初始化原始模型 (PureDecoderTransformer) 使用infer_id_embs_fn
    ori_model_with_fn = PureDecoderTransformer(
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
        lm_head=None,
        infer_id_embs_fn=infer_id_embs_fn
    )
    
    # 初始化新模型 (TokenDecoder) 使用infer_id_embs_fn
    new_model_with_fn = TokenDecoder(
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
        lm_head=None,
        infer_id_embs_fn=infer_id_embs_fn,
        attention_function=config["attention_function"]
    )
    
    # 设置为评估模式
    ori_model_with_fn.eval()
    new_model_with_fn.eval()
    
    # 禁用梯度计算
    with torch.no_grad():
        # 获取原始模型的输出
        ori_output_with_fn = ori_model_with_fn.forward_with_tokens(test_input_ids)
        print(f"原始模型(有infer_id_embs_fn)输出形状: {ori_output_with_fn.shape}")
        
        # 获取新模型的输出
        new_output_with_fn = new_model_with_fn.forward_with_tokens(test_input_ids)
        print(f"新模型(有infer_id_embs_fn)输出形状: {new_output_with_fn.shape}")
        
        # 计算输出差异
        diff_with_fn = torch.abs(ori_output_with_fn - new_output_with_fn)
        max_diff_with_fn = torch.max(diff_with_fn)
        mean_diff_with_fn = torch.mean(diff_with_fn)
        
        print(f"有infer_id_embs_fn时输出最大差异: {max_diff_with_fn.item()}")
        print(f"有infer_id_embs_fn时输出平均差异: {mean_diff_with_fn.item()}")
        
        # 检查差异是否在可接受范围内
        consistency_with_fn = max_diff_with_fn < 1e-5 and mean_diff_with_fn < 1e-6
        print(f"有infer_id_embs_fn时一致性测试: {'✅ 通过' if consistency_with_fn else '❌ 失败'}")
    
    # 总体结果
    overall_success = consistency_no_fn and consistency_with_fn
    print(f"\n总体测试结果: {'✅ 全部通过' if overall_success else '❌ 存在失败'}")
    return overall_success


if __name__ == "__main__":
    success = test_infer_id_embs_consistency()
    sys.exit(0 if success else 1)