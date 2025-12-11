import torch
import torch.nn as nn
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# 导入两个模型类
from muse.models.keye_ar.token_decoder_ori import PureDecoderTransformer
from muse.models.keye_ar.token_decoder import TokenDecoder


class LayerComparisonDebugger:
    def __init__(self, ori_model, new_model):
        self.ori_model = ori_model
        self.new_model = new_model
        self.ori_outputs = {}
        self.new_outputs = {}
        self.hooks = []
        
    def register_hooks(self):
        """注册forward hooks来捕获每层的输出"""
        # 清除之前的hooks和outputs
        self.clear_hooks()
        self.ori_outputs.clear()
        self.new_outputs.clear()
        
        # 为原始模型注册hooks
        self._register_model_hooks(self.ori_model, "ori", self.ori_outputs)
        
        # 为新模型注册hooks
        self._register_model_hooks(self.new_model, "new", self.new_outputs)
        
    def _register_model_hooks(self, model, prefix, output_dict):
        """为模型注册hooks"""
        # 注册token embedding层的hook
        if hasattr(model, 'token_embedding'):
            hook = model.token_embedding.register_forward_hook(
                lambda module, input, output: output_dict.update({f"{prefix}_token_embedding": output.clone()})
            )
            self.hooks.append(hook)
            
        # 注册position embedding层的hook
        if hasattr(model, 'position_embedding'):
            hook = model.position_embedding.register_forward_hook(
                lambda module, input, output: output_dict.update({f"{prefix}_position_embedding": output.clone()})
            )
            self.hooks.append(hook)
            
        # 注册input_linear层的hook
        if hasattr(model, 'input_linear'):
            hook = model.input_linear.register_forward_hook(
                lambda module, input, output: output_dict.update({f"{prefix}_input_linear": output.clone()})
            )
            self.hooks.append(hook)
            
        # 注册各层decoder的hooks
        if hasattr(model, 'layers'):
            for i, layer in enumerate(model.layers):
                hook = layer.register_forward_hook(
                    lambda module, input, output, idx=i: output_dict.update({f"{prefix}_layer_{idx}": output.clone()})
                )
                self.hooks.append(hook)
                
        # 注册final_norm层的hook（如果是原始模型）
        if hasattr(model, 'final_norm'):
            hook = model.final_norm.register_forward_hook(
                lambda module, input, output: output_dict.update({f"{prefix}_final_norm": output.clone()})
            )
            self.hooks.append(hook)
            
        # 注册output_linear层的hook
        if hasattr(model, 'output_linear'):
            hook = model.output_linear.register_forward_hook(
                lambda module, input, output: output_dict.update({f"{prefix}_output_linear": output.clone()})
            )
            self.hooks.append(hook)
            
        # 注册lm_head层的hook（如果存在）
        if hasattr(model, 'lm_head') and model.lm_head is not None:
            hook = model.lm_head.register_forward_hook(
                lambda module, input, output: output_dict.update({f"{prefix}_lm_head": output.clone()})
            )
            self.hooks.append(hook)
            
        # 对于新模型，还需要注册transformer内部层的hooks
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'layers'):
            for i, layer in enumerate(model.transformer.layers):
                hook = layer.register_forward_hook(
                    lambda module, input, output, idx=i: output_dict.update({f"{prefix}_transformer_layer_{idx}": output.clone()})
                )
                self.hooks.append(hook)
                
    def clear_hooks(self):
        """清除所有hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        
    def compare_outputs(self):
        """比较两个模型各层的输出"""
        print("\n=== 各层输出比较 ===")
        
        # 获取所有层名
        all_keys = set(self.ori_outputs.keys()) | set(self.new_outputs.keys())
        
        for key in sorted(all_keys):
            # 移除前缀以进行比较
            ori_key = key.replace("new_", "ori_")
            new_key = key.replace("ori_", "new_")
            
            if ori_key in self.ori_outputs and new_key in self.new_outputs:
                ori_output = self.ori_outputs[ori_key]
                new_output = self.new_outputs[new_key]
                
                # 计算差异
                diff = torch.abs(ori_output - new_output)
                max_diff = torch.max(diff).item()
                mean_diff = torch.mean(diff).item()
                
                print(f"{key}:")
                print(f"  形状: {ori_output.shape}")
                print(f"  最大差异: {max_diff:.6f}")
                print(f"  平均差异: {mean_diff:.6f}")
                
                # 如果差异较大，打印详细信息
                if max_diff > 1e-5:
                    print(f"  ⚠️  差异较大!")
                    
                    # 打印前几个元素的值
                    flat_ori = ori_output.flatten()
                    flat_new = new_output.flatten()
                    print(f"  原始模型前5个值: {[f'{x:.6f}' for x in flat_ori[:5].tolist()]}")
                    print(f"  新模型前5个值: {[f'{x:.6f}' for x in flat_new[:5].tolist()]}")
            else:
                print(f"{key}: 只在一个模型中存在")
                if ori_key in self.ori_outputs:
                    print(f"  原始模型输出形状: {self.ori_outputs[ori_key].shape}")
                if new_key in self.new_outputs:
                    print(f"  新模型输出形状: {self.new_outputs[new_key].shape}")
                    
    def debug_forward_pass(self, input_embeddings):
        """执行前向传播并比较各层输出"""
        print("\n=== 开始调试前向传播 ===")
        
        # 注册hooks
        self.register_hooks()
        
        # 设置为评估模式
        self.ori_model.eval()
        self.new_model.eval()
        
        # 禁用梯度计算
        with torch.no_grad():
            print("执行原始模型前向传播...")
            ori_output = self.ori_model(input_embeddings)
            
            print("执行新模型前向传播...")
            new_output = self.new_model(input_embeddings)
            
            # 比较最终输出
            diff = torch.abs(ori_output - new_output)
            max_diff = torch.max(diff).item()
            mean_diff = torch.mean(diff).item()
            
            print(f"\n=== 最终输出比较 ===")
            print(f"原始模型输出形状: {ori_output.shape}")
            print(f"新模型输出形状: {new_output.shape}")
            print(f"输出最大差异: {max_diff:.6f}")
            print(f"输出平均差异: {mean_diff:.6f}")
            
            # 比较各层输出
            self.compare_outputs()
            
        # 清除hooks
        self.clear_hooks()


def debug_convert_hf_state_dict():
    """
    调试convert_hf_state_dict函数的正确性
    """
    print("开始调试convert_hf_state_dict函数...")
    
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
    
    # 5. 创建调试器并执行调试
    print("\n创建调试器...")
    debugger = LayerComparisonDebugger(ori_model, new_model)
    
    # 创建测试输入
    batch_size = 2
    seq_len = 5
    input_embeddings = torch.randn(batch_size, seq_len, config["d_model"])
    
    # 执行调试前向传播
    debugger.debug_forward_pass(input_embeddings)


if __name__ == "__main__":
    debug_convert_hf_state_dict()