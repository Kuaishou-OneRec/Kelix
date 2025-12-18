import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from token_decoder_ori import PureDecoderTransformer
from muse.models.keye_ar.unified_token_decoder import TokenDecoder

class LayerComparisonDebugger:
    def __init__(self, ori_model: PureDecoderTransformer, new_model: TokenDecoder):
        self.ori_model = ori_model
        self.new_model = new_model
        self.ori_outputs = {}
        self.new_outputs = {}
        
        # 注册hook以捕获各层输出
        self._register_hooks()
    
    def _register_hooks(self):
        """注册forward hooks以捕获各层输出"""
        # 清空之前的记录
        self.ori_outputs.clear()
        self.new_outputs.clear()
        
        # 原始模型hooks
        self.ori_model.token_embedding.register_forward_hook(
            lambda module, input, output: self.ori_outputs.update({"token_embedding": output})
        )
        self.ori_model.position_embedding.register_forward_hook(
            lambda module, input, output: self.ori_outputs.update({"position_embedding": output})
        )
        self.ori_model.input_linear.register_forward_hook(
            lambda module, input, output: self.ori_outputs.update({"input_linear": output})
        )
        
        # 注册原始模型的transformer层hooks
        for i, layer in enumerate(self.ori_model.layers):
            layer.register_forward_hook(
                lambda module, input, output, idx=i: self.ori_outputs.update({f"layer_{idx}": output})
            )
        
        self.ori_model.output_linear.register_forward_hook(
            lambda module, input, output: self.ori_outputs.update({"output_linear": output})
        )
        
        # 新模型hooks
        self.new_model.token_embedding.register_forward_hook(
            lambda module, input, output: self.new_outputs.update({"token_embedding": output})
        )
        self.new_model.position_embedding.register_forward_hook(
            lambda module, input, output: self.new_outputs.update({"position_embedding": output})
        )
        self.new_model.input_linear.register_forward_hook(
            lambda module, input, output: self.new_outputs.update({"input_linear": output})
        )
        
        # 注册新模型的transformer层hooks
        for i, layer in enumerate(self.new_model.transformer.layers):
            layer.register_forward_hook(
                lambda module, input, output, idx=i: self.new_outputs.update({f"layer_{idx}": output})
            )
        
        self.new_model.output_linear.register_forward_hook(
            lambda module, input, output: self.new_outputs.update({"output_linear": output})
        )
    
    def compare_outputs(self) -> Dict[str, Dict[str, Any]]:
        """比较各层输出并返回差异分析"""
        comparison_results = {}
        
        # 收集所有层名称
        all_layers = set(self.ori_outputs.keys()) | set(self.new_outputs.keys())
        
        for layer_name in all_layers:
            result = {
                "ori_shape": None,
                "new_shape": None,
                "max_diff": None,
                "mean_diff": None,
                "ori_values": None,
                "new_values": None
            }
            
            ori_output = self.ori_outputs.get(layer_name)
            new_output = self.new_outputs.get(layer_name)
            
            if ori_output is not None and new_output is not None:
                # 两模型都有的层
                result["ori_shape"] = ori_output.shape
                result["new_shape"] = new_output.shape
                
                # 计算差异
                diff = torch.abs(ori_output - new_output)
                result["max_diff"] = diff.max().item()
                result["mean_diff"] = diff.mean().item()
                
                # 显示前5个值用于调试
                result["ori_values"] = [f"{v:.6f}" for v in ori_output.flatten()[:5].tolist()]
                result["new_values"] = [f"{v:.6f}" for v in new_output.flatten()[:5].tolist()]
            elif ori_output is not None:
                # 只在原始模型中存在的层
                result["ori_shape"] = ori_output.shape
                result["new_shape"] = "N/A"
                result["max_diff"] = "Only in original model"
            else:
                # 只在新模型中存在的层
                result["ori_shape"] = "N/A"
                result["new_shape"] = new_output.shape
                result["max_diff"] = "Only in new model"
            
            comparison_results[layer_name] = result
        
        return comparison_results
    
    def print_comparison(self):
        """打印比较结果"""
        results = self.compare_outputs()
        
        print("\n=== 各层输出比较 ===")
        # 按照特定顺序排序显示
        layer_order = [
            "token_embedding",
            "position_embedding", 
            "input_linear",
            "layer_0",
            "layer_1",
            "output_linear"
        ]
        
        # 先显示按顺序排列的层
        for layer_name in layer_order:
            if layer_name in results:
                result = results[layer_name]
                print(f"{layer_name}:")
                print(f"  原始模型形状: {result['ori_shape']}")
                print(f"  新模型形状: {result['new_shape']}")
                
                if isinstance(result['max_diff'], (int, float)):
                    print(f"  最大差异: {result['max_diff']:.6f}")
                    print(f"  平均差异: {result['mean_diff']:.6f}")
                    if result['max_diff'] > 1e-5:  # 如果差异较大
                        print("  ⚠️  差异较大!")
                    print(f"  原始模型前5个值: {result['ori_values']}")
                    print(f"  新模型前5个值: {result['new_values']}")
                else:
                    print(f"  {result['max_diff']}")
                print()
        
        # 显示其他层
        for layer_name, result in results.items():
            if layer_name not in layer_order:
                print(f"{layer_name}:")
                print(f"  原始模型形状: {result['ori_shape']}")
                print(f"  新模型形状: {result['new_shape']}")
                
                if isinstance(result['max_diff'], (int, float)):
                    print(f"  最大差异: {result['max_diff']:.6f}")
                    print(f"  平均差异: {result['mean_diff']:.6f}")
                    if result['max_diff'] > 1e-5:  # 如果差异较大
                        print("  ⚠️  差异较大!")
                    print(f"  原始模型前5个值: {result['ori_values']}")
                    print(f"  新模型前5个值: {result['new_values']}")
                else:
                    print(f"  {result['max_diff']}")
                print()

def debug_convert_hf_state_dict():
    """调试状态字典转换过程"""
    # 配置参数（与test_convert_hf_state_dict.py保持一致）
    vocab_size = 1000
    max_length = 30
    d_model = 128
    eos_token = 999
    nhead = 4
    num_layers = 2
    dim_feedforward = 512
    reduce = False  # 关键参数
    
    print(f"配置参数: d_model={d_model}, nhead={nhead}, num_layers={num_layers}, reduce={reduce}")
    
    # 必须手动创建token_embedding以在两个模型间共享
    token_embedding = nn.Embedding(vocab_size, d_model)
    
    # 创建原始模型
    ori_model = PureDecoderTransformer(
        vocab_size=vocab_size,
        max_length=max_length,
        d_model=d_model,
        eos_token=eos_token,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        token_embedding=token_embedding,  # 共享token_embedding
        use_flash_attn=False,
        use_gradient_checkpointing=False,
        reduce=reduce
    )
    
    # 创建新模型
    new_model = TokenDecoder(
        vocab_size=vocab_size,
        max_length=max_length,
        d_model=d_model,
        eos_token=eos_token,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        token_embedding=token_embedding,  # 共享token_embedding
        reduce=reduce
    )
    
    # 获取原始模型状态字典
    ori_state_dict = ori_model.state_dict()
    print(f"原始模型状态字典键数: {len(ori_state_dict)}")
    
    # 转换状态字典
    converted_state_dict = TokenDecoder.convert_hf_state_dict(ori_state_dict, reduce_mode=reduce)
    print(f"转换后模型状态字典键数: {len(converted_state_dict)}")
    
    # 加载转换后的状态字典到新模型
    print("\n加载转换后的状态字典到新模型...")
    new_model.load_state_dict(converted_state_dict, strict=False)
    
    # 创建调试器
    print("\n创建调试器...")
    debugger = LayerComparisonDebugger(ori_model, new_model)
    
    # 准备测试输入
    batch_size, seq_len = 2, 5
    test_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print("\n=== 开始调试前向传播 ===")
    # 执行前向传播 - 使用forward_with_tokens方法
    print("执行原始模型前向传播...")
    with torch.no_grad():
        ori_output = ori_model.forward_with_tokens(test_input)
    
    print("执行新模型前向传播...")
    with torch.no_grad():
        new_output = new_model.forward_with_tokens(test_input)
    
    # 比较最终输出
    print("\n=== 最终输出比较 ===")
    print(f"原始模型输出形状: {ori_output.shape}")
    print(f"新模型输出形状: {new_output.shape}")
    
    diff = torch.abs(ori_output - new_output)
    print(f"输出最大差异: {diff.max().item():.6f}")
    print(f"输出平均差异: {diff.mean().item():.6f}")
    
    # 打印详细比较
    debugger.print_comparison()

if __name__ == "__main__":
    debug_convert_hf_state_dict()