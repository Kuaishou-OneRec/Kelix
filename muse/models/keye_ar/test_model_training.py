import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# 导入新模型类
from muse.models.keye_ar.token_decoder import TokenDecoder


def test_model_training():
    """
    测试新模型的训练功能
    """
    print("开始测试模型训练功能...")
    
    # 1. 设置配置参数
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
    
    # 创建LM Head用于训练，注意维度顺序：从d_model到vocab_size
    lm_head = nn.Linear(config["d_model"], config["vocab_size"])
    
    # 2. 初始化模型，将lm_head作为参数传入
    print("\n初始化模型...")
    model = TokenDecoder(
        vocab_size=config["vocab_size"],
        max_length=config["max_length"],
        d_model=config["d_model"],
        eos_token=config["eos_token"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        dim_feedforward=config["dim_feedforward"],
        use_gradient_checkpointing=config["use_gradient_checkpointing"],
        reduce=config["reduce"],
        attention_function=config["attention_function"],
        lm_head=lm_head  # 将lm_head作为参数传入
    )
    
    # 3. 创建优化器和损失函数
    print("\n创建优化器和损失函数...")
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # 4. 创建模拟数据
    print("\n创建模拟训练数据...")
    batch_size = 4
    seq_len = 10
    num_steps = 5
    
    # 生成随机训练数据
    train_data = torch.randint(0, config["vocab_size"], (batch_size, seq_len))
    
    # 5. 训练几个步骤并观察损失下降
    print("\n开始训练测试...")
    model.train()
    
    losses = []
    for step in range(num_steps):
        # 构造输入和目标
        input_ids = train_data[:, :-1]  # 所有token除了最后一个
        target_ids = train_data[:, 1:]  # 所有token除了第一个（预测下一个token）
        
        # 前向传播
        # 对于reduce=False模式，forward_with_tokens的输出已经是最终的logits
        logits = model.forward_with_tokens(input_ids)
        
        # 确保logits的形状正确
        # logits应该已经是(Batch, Seq_Len, vocab_size)
        assert logits.shape[-1] == config["vocab_size"], f"logits最后一维应该是{config['vocab_size']}，但实际是{logits.shape[-1]}"
        
        # 计算损失
        loss = criterion(
            logits.reshape(-1, config["vocab_size"]), 
            target_ids.reshape(-1)
        )
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        print(f"Step {step+1}/{num_steps}, Loss: {loss.item():.6f}")
    
    # 6. 验证损失是否下降
    print(f"\n初始损失: {losses[0]:.6f}")
    print(f"最终损失: {losses[-1]:.6f}")
    
    # 检查损失是否显著下降
    loss_decrease = losses[0] - losses[-1]
    print(f"损失下降: {loss_decrease:.6f}")
    
    # 判断训练是否有效
    training_success = loss_decrease > 0.1  # 损失下降超过0.1认为训练有效
    print(f"训练有效性测试: {'✅ 通过' if training_success else '❌ 失败'}")
    
    # 7. 验证梯度是否存在
    print("\n验证梯度存在性...")
    gradients_exist = False
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.norm() > 0:
            gradients_exist = True
            print(f"✓ 参数 {name} 存在有效梯度")
            break
    
    if not gradients_exist:
        print("✗ 未检测到有效梯度")
    
    overall_success = training_success and gradients_exist
    print(f"\n总体训练测试结果: {'✅ 通过' if overall_success else '❌ 失败'}")
    
    return overall_success


if __name__ == "__main__":
    success = test_model_training()
    sys.exit(0 if success else 1)