import torch
import torch.nn as nn
import sys
import os
import pytest

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# 导入模型类
from muse.models.keye_ar.token_decoder_ori import PureDecoderTransformer
from muse.models.keye_ar.unified_token_decoder import TokenDecoder


class TestLayerComparison:
    """测试模型层输出比较功能"""
    
    def setup_method(self):
        """测试初始化配置"""
        self.config = {
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
        
        # 创建共享的token embedding
        self.token_embedding = nn.Embedding(self.config["vocab_size"], self.config["d_model"])
        
        # 初始化原始模型
        self.ori_model = PureDecoderTransformer(
            vocab_size=self.config["vocab_size"],
            max_length=self.config["max_length"],
            d_model=self.config["d_model"],
            eos_token=self.config["eos_token"],
            nhead=self.config["nhead"],
            num_layers=self.config["num_layers"],
            dim_feedforward=self.config["dim_feedforward"],
            token_embedding=self.token_embedding,
            use_flash_attn=False,
            use_gradient_checkpointing=self.config["use_gradient_checkpointing"],
            reduce=self.config["reduce"],
            lm_head=None
        )
        
        # 初始化新模型
        self.new_model = TokenDecoder(
            vocab_size=self.config["vocab_size"],
            max_length=self.config["max_length"],
            d_model=self.config["d_model"],
            eos_token=self.config["eos_token"],
            nhead=self.config["nhead"],
            num_layers=self.config["num_layers"],
            dim_feedforward=self.config["dim_feedforward"],
            token_embedding=self.token_embedding,
            use_gradient_checkpointing=self.config["use_gradient_checkpointing"],
            reduce=self.config["reduce"],
            attention_function=self.config["attention_function"]
        )

    def test_convert_hf_state_dict(self):
        """测试convert_hf_state_dict函数的正确性"""
        print("开始测试convert_hf_state_dict函数...")
        
        # 转换状态字典
        ori_state_dict = self.ori_model.state_dict()
        print(f"原始模型状态字典键数: {len(ori_state_dict)}")
        
        # 使用新模型的convert_hf_state_dict方法转换状态字典
        converted_state_dict = TokenDecoder.convert_hf_state_dict(ori_state_dict)
        print(f"转换后模型状态字典键数: {len(converted_state_dict)}")
        
        # 将转换后的状态字典加载到新模型中
        self.new_model.load_state_dict(converted_state_dict)
        
        # 比较两个模型的前向结果
        batch_size = 2
        seq_len = 5
        input_embeddings = torch.randn(batch_size, seq_len, self.config["d_model"])
        
        # 设置为评估模式
        self.ori_model.eval()
        self.new_model.eval()
        
        # 禁用梯度计算
        with torch.no_grad():
            # 获取原始模型的输出
            ori_output = self.ori_model(input_embeddings)
            print(f"原始模型输出形状: {ori_output.shape}")
            
            # 获取新模型的输出
            new_output = self.new_model(input_embeddings)
            print(f"新模型输出形状: {new_output.shape}")
            
            # 计算输出差异
            diff = torch.abs(ori_output - new_output)
            max_diff = torch.max(diff)
            mean_diff = torch.mean(diff)
            
            print(f"输出最大差异: {max_diff.item()}")
            print(f"输出平均差异: {mean_diff.item()}")
            
            # 检查差异是否在可接受范围内
            assert max_diff < 1e-5 and mean_diff < 1e-6, "两个模型的前向结果不一致"


class TestInferIdEmbsConsistency:
    """测试infer_id_embs_fn一致性"""
    
    def setup_method(self):
        """测试初始化配置"""
        self.config = {
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
        
        # 创建共享的token embedding和测试输入
        self.token_embedding = nn.Embedding(self.config["vocab_size"], self.config["d_model"])
        
        # 创建测试输入
        batch_size = 2
        seq_len = 5
        self.test_input_ids = torch.randint(0, self.config["vocab_size"], (batch_size, seq_len))

    def test_infer_id_embs_consistency_without_fn(self):
        """测试不给定infer_id_embs_fn时的一致性"""
        print("\n=== 测试1: 不给定infer_id_embs_fn时的一致性 ===")
        
        # 初始化原始模型 (PureDecoderTransformer) 不使用infer_id_embs_fn
        ori_model_no_fn = PureDecoderTransformer(
            vocab_size=self.config["vocab_size"],
            max_length=self.config["max_length"],
            d_model=self.config["d_model"],
            eos_token=self.config["eos_token"],
            nhead=self.config["nhead"],
            num_layers=self.config["num_layers"],
            dim_feedforward=self.config["dim_feedforward"],
            token_embedding=self.token_embedding,
            use_flash_attn=False,
            use_gradient_checkpointing=self.config["use_gradient_checkpointing"],
            reduce=self.config["reduce"],
            lm_head=None,
            infer_id_embs_fn=None
        )
        
        # 获取原始模型的状态字典
        ori_state_dict = ori_model_no_fn.state_dict()
        
        # 使用TokenDecoder的convert_hf_state_dict方法转换状态字典
        converted_state_dict = TokenDecoder.convert_hf_state_dict(ori_state_dict, reduce_mode=self.config["reduce"])
        
        # 初始化新模型 (TokenDecoder) 不使用infer_id_embs_fn
        new_model_no_fn = TokenDecoder(
            vocab_size=self.config["vocab_size"],
            max_length=self.config["max_length"],
            d_model=self.config["d_model"],
            eos_token=self.config["eos_token"],
            nhead=self.config["nhead"],
            num_layers=self.config["num_layers"],
            dim_feedforward=self.config["dim_feedforward"],
            token_embedding=self.token_embedding,
            use_gradient_checkpointing=self.config["use_gradient_checkpointing"],
            reduce=self.config["reduce"],
            lm_head=None,
            infer_id_embs_fn=None,
            attention_function=self.config["attention_function"]
        )
        
        # 将转换后的状态字典加载到新模型中
        new_model_no_fn.load_state_dict(converted_state_dict)
        
        # 设置为评估模式
        ori_model_no_fn.eval()
        new_model_no_fn.eval()
        
        # 禁用梯度计算
        with torch.no_grad():
            # 获取原始模型的输出
            ori_output_no_fn = ori_model_no_fn.forward_with_tokens(self.test_input_ids)
            print(f"原始模型(无infer_id_embs_fn)输出形状: {ori_output_no_fn.shape}")
            
            # 获取新模型的输出
            new_output_no_fn = new_model_no_fn.forward_with_tokens(self.test_input_ids)
            print(f"新模型(无infer_id_embs_fn)输出形状: {new_output_no_fn.shape}")
            
            # 计算输出差异
            diff_no_fn = torch.abs(ori_output_no_fn - new_output_no_fn)
            max_diff_no_fn = torch.max(diff_no_fn)
            mean_diff_no_fn = torch.mean(diff_no_fn)
            
            print(f"无infer_id_embs_fn时输出最大差异: {max_diff_no_fn.item()}")
            print(f"无infer_id_embs_fn时输出平均差异: {mean_diff_no_fn.item()}")
            
            # 检查差异是否在可接受范围内
            assert max_diff_no_fn < 1e-5 and mean_diff_no_fn < 1e-6, "无infer_id_embs_fn时不一致"

    def test_infer_id_embs_consistency_with_fn(self):
        """测试给定infer_id_embs_fn时的一致性"""
        print("\n=== 测试2: 给定infer_id_embs_fn时的一致性 ===")
        
        # 定义一个简单的infer_id_embs_fn示例
        def infer_id_embs_fn(ids):
            # 使用标准embedding，但在实际应用中可能有更复杂的逻辑
            return self.token_embedding(ids)
        
        # 初始化原始模型 (PureDecoderTransformer) 使用infer_id_embs_fn
        ori_model_with_fn = PureDecoderTransformer(
            vocab_size=self.config["vocab_size"],
            max_length=self.config["max_length"],
            d_model=self.config["d_model"],
            eos_token=self.config["eos_token"],
            nhead=self.config["nhead"],
            num_layers=self.config["num_layers"],
            dim_feedforward=self.config["dim_feedforward"],
            token_embedding=self.token_embedding,
            use_flash_attn=False,
            use_gradient_checkpointing=self.config["use_gradient_checkpointing"],
            reduce=self.config["reduce"],
            lm_head=None,
            infer_id_embs_fn=infer_id_embs_fn
        )
        
        # 获取原始模型的状态字典
        ori_state_dict_with_fn = ori_model_with_fn.state_dict()
        
        # 使用TokenDecoder的convert_hf_state_dict方法转换状态字典
        converted_state_dict_with_fn = TokenDecoder.convert_hf_state_dict(ori_state_dict_with_fn, reduce_mode=self.config["reduce"])
        
        # 初始化新模型 (TokenDecoder) 使用infer_id_embs_fn
        new_model_with_fn = TokenDecoder(
            vocab_size=self.config["vocab_size"],
            max_length=self.config["max_length"],
            d_model=self.config["d_model"],
            eos_token=self.config["eos_token"],
            nhead=self.config["nhead"],
            num_layers=self.config["num_layers"],
            dim_feedforward=self.config["dim_feedforward"],
            token_embedding=self.token_embedding,
            use_gradient_checkpointing=self.config["use_gradient_checkpointing"],
            reduce=self.config["reduce"],
            lm_head=None,
            infer_id_embs_fn=infer_id_embs_fn,
            attention_function=self.config["attention_function"]
        )
        
        # 将转换后的状态字典加载到新模型中
        new_model_with_fn.load_state_dict(converted_state_dict_with_fn)
        
        # 设置为评估模式
        ori_model_with_fn.eval()
        new_model_with_fn.eval()
        
        # 禁用梯度计算
        with torch.no_grad():
            # 获取原始模型的输出
            ori_output_with_fn = ori_model_with_fn.forward_with_tokens(self.test_input_ids)
            print(f"原始模型(有infer_id_embs_fn)输出形状: {ori_output_with_fn.shape}")
            
            # 获取新模型的输出
            new_output_with_fn = new_model_with_fn.forward_with_tokens(self.test_input_ids)
            print(f"新模型(有infer_id_embs_fn)输出形状: {new_output_with_fn.shape}")
            
            # 计算输出差异
            diff_with_fn = torch.abs(ori_output_with_fn - new_output_with_fn)
            max_diff_with_fn = torch.max(diff_with_fn)
            mean_diff_with_fn = torch.mean(diff_with_fn)
            
            print(f"有infer_id_embs_fn时输出最大差异: {max_diff_with_fn.item()}")
            print(f"有infer_id_embs_fn时输出平均差异: {mean_diff_with_fn.item()}")
            
            # 检查差异是否在可接受范围内
            assert max_diff_with_fn < 1e-5 and mean_diff_with_fn < 1e-6, "有infer_id_embs_fn时不一致"


class TestRevertHfStateDict:
    """测试revert_hf_state_dict函数对称性"""
    
    def setup_method(self):
        """测试初始化配置"""
        self.config = {
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
        
        # 初始化原始模型
        self.token_embedding = nn.Embedding(self.config["vocab_size"], self.config["d_model"])
        
        self.ori_model = PureDecoderTransformer(
            vocab_size=self.config["vocab_size"],
            max_length=self.config["max_length"],
            d_model=self.config["d_model"],
            eos_token=self.config["eos_token"],
            nhead=self.config["nhead"],
            num_layers=self.config["num_layers"],
            dim_feedforward=self.config["dim_feedforward"],
            token_embedding=self.token_embedding,
            use_flash_attn=False,
            use_gradient_checkpointing=self.config["use_gradient_checkpointing"],
            reduce=self.config["reduce"],
            lm_head=None
        )

    def test_revert_hf_state_dict_symmetry(self):
        """测试revert_hf_state_dict函数是否与convert_hf_state_dict完全对称"""
        print("开始测试revert_hf_state_dict对称性...")
        
        # 获取原始模型状态字典
        ori_state_dict = self.ori_model.state_dict()
        print(f"原始模型状态字典键数: {len(ori_state_dict)}")
        
        # 转换为新模型状态字典
        converted_state_dict = TokenDecoder.convert_hf_state_dict(ori_state_dict, reduce_mode=self.config["reduce"])
        print(f"转换后模型状态字典键数: {len(converted_state_dict)}")
        
        # 还原回原始模型状态字典
        reverted_state_dict = TokenDecoder.revert_hf_state_dict(converted_state_dict, reduce_mode=self.config["reduce"])
        print(f"还原后模型状态字典键数: {len(reverted_state_dict)}")
        
        # 检查键的数量是否一致
        keys_match = set(ori_state_dict.keys()) == set(reverted_state_dict.keys())
        assert keys_match, "键集合不一致"
        
        # 检查每个键对应的张量是否一致
        tensors_match = True
        max_diff = 0.0
        
        for key in ori_state_dict.keys():
            if key in reverted_state_dict:
                ori_tensor = ori_state_dict[key]
                rev_tensor = reverted_state_dict[key]
                
                # 检查形状是否一致
                assert ori_tensor.shape == rev_tensor.shape, f"键 {key} 形状不一致: 原始 {ori_tensor.shape} vs 还原 {rev_tensor.shape}"
                
                # 计算差异
                diff = torch.abs(ori_tensor - rev_tensor)
                max_diff_tensor = torch.max(diff).item()
                mean_diff_tensor = torch.mean(diff).item()
                
                if max_diff_tensor > max_diff:
                    max_diff = max_diff_tensor
                
                # 检查差异是否在可接受范围内
                if max_diff_tensor > 1e-6 or mean_diff_tensor > 1e-7:
                    tensors_match = False
                    break
            else:
                tensors_match = False
                break
        
        assert tensors_match, "张量形状和数值不一致"
        
        # 测试转换-还原-再转换的对称性
        # 再次转换还原后的状态字典
        double_converted_state_dict = TokenDecoder.convert_hf_state_dict(reverted_state_dict, reduce_mode=self.config["reduce"])
        
        # 比较第一次转换和第二次转换的结果
        conversion_symmetric = True
        conversion_max_diff = 0.0
        
        for key in converted_state_dict.keys():
            if key in double_converted_state_dict:
                first_conversion = converted_state_dict[key]
                second_conversion = double_converted_state_dict[key]
                
                # 检查形状是否一致
                assert first_conversion.shape == second_conversion.shape, f"二次转换键 {key} 形状不一致"
                
                # 计算差异
                diff = torch.abs(first_conversion - second_conversion)
                max_diff_tensor = torch.max(diff).item()
                
                if max_diff_tensor > conversion_max_diff:
                    conversion_max_diff = max_diff_tensor
                
                # 检查差异是否在可接受范围内
                if max_diff_tensor > 1e-6:
                    conversion_symmetric = False
                    break
            else:
                conversion_symmetric = False
                break
        
        assert conversion_symmetric, "转换-还原-再转换不对称"


class TestModelTraining:
    """测试模型训练功能"""
    
    def test_model_training(self):
        """测试新模型的训练功能"""
        print("开始测试模型训练功能...")
        
        # 设置配置参数
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
        
        # 初始化模型，将lm_head作为参数传入
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
        
        # 创建优化器和损失函数
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        # 创建模拟数据
        batch_size = 4
        seq_len = 10
        num_steps = 5
        
        # 生成随机训练数据
        train_data = torch.randint(0, config["vocab_size"], (batch_size, seq_len))
        
        # 训练几个步骤并观察损失下降
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
        
        # 验证损失是否下降
        print(f"\n初始损失: {losses[0]:.6f}")
        print(f"最终损失: {losses[-1]:.6f}")
        
        # 检查损失是否显著下降
        loss_decrease = losses[0] - losses[-1]
        print(f"损失下降: {loss_decrease:.6f}")
        
        # 判断训练是否有效
        assert loss_decrease > 0.1, "训练未能有效降低损失"
        
        # 验证梯度是否存在
        print("\n验证梯度存在性...")
        gradients_exist = False
        for name, param in model.named_parameters():
            if param.grad is not None and param.grad.norm() > 0:
                gradients_exist = True
                print(f"✓ 参数 {name} 存在有效梯度")
                break
        
        assert gradients_exist, "未检测到有效梯度"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])