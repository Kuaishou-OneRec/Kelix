from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
import sys


model_name1 = sys.argv[1]
model_name2 = sys.argv[2]
# model_name1 = "/mmu_mllm_hdd_2/lingzhixin/models/Keye-32B-vit0.8.1_0606"
# model_name2 = "/mmu_mllm_hdd_2/lingzhixin/models/Keye-32B-scratch_0606"

# python3 /llm_reco/lingzhixin/recovlm_qw0510/recovlm/tools/model_helpers/hf_converters/compare_ckpt.py /mmu_mllm_hdd_2/lingzhixin/models/Keye-8B-demo_hf_vit_rope_nopos_qknorm_0714_v2/ /mmu_mllm_hdd_2/lingzhixin/models/Keye-8B-demo_hf_vit_rope_nopos_0714_v1/

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name1, trust_remote_code=True)
model1 = AutoModelForCausalLM.from_pretrained(
    model_name1,
    _attn_implementation = 'flash_attention_2',
    torch_dtype="auto",
    device_map="cpu",
    trust_remote_code=True,
    ignore_mismatched_sizes=True
)


# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name2, trust_remote_code=True)
model2 = AutoModelForCausalLM.from_pretrained(
    model_name2,
    _attn_implementation = 'flash_attention_2',
    torch_dtype="auto",
    device_map="cpu",
    trust_remote_code=True,
    ignore_mismatched_sizes=True
)





import torch

def compare_model_parameters(model1, model2, threshold=1e-6):
    """
    比较两个模型的参数，找出不同的权重
    
    参数:
    model1, model2: 要比较的两个PyTorch模型
    threshold: 认为参数不同的最小绝对差异值
    
    返回:
    different_params: 包含不同参数名称和差异详情的字典
    """
    different_params = {}
    
    # 获取两个模型的所有参数
    params1 = dict(model1.named_parameters())
    params2 = dict(model2.named_parameters())
    
    # 确保两个模型参数名称完全一致
    param_names1 = set(params1.keys())
    param_names2 = set(params2.keys())
    
    if param_names1 != param_names2:
        print("警告: 两个模型的参数名称不完全一致")
        missing_in_1 = param_names2 - param_names1
        missing_in_2 = param_names1 - param_names2
        if missing_in_1:
            print(f"在model2中存在但model1中缺失的参数: {missing_in_1}")
        if missing_in_2:
            print(f"在model1中存在但model2中缺失的参数: {missing_in_2}")
    
    # 比较共同参数
    common_params = param_names1 & param_names2
    total_params = len(common_params)
    different_count = 0
    
    for name in common_params:
        param1 = params1[name]
        param2 = params2[name]
        
        # 计算参数差异
        diff = torch.abs(param1 - param2)
        max_diff = diff.max().item()
        
        # 如果差异超过阈值，则记录
        if max_diff > threshold:
            different_params[name] = {
                'max_diff': max_diff,
                'mean_diff': diff.mean().item(),
                'shape': tuple(param1.shape),
                'param1_min': param1.min().item(),
                'param1_max': param1.max().item(),
                'param2_min': param2.min().item(),
                'param2_max': param2.max().item()
            }
            different_count += 1
    
    # 输出比较结果统计
    print(f"参数比较完成:")
    print(f"- 总共比较参数: {total_params}")
    print(f"- 不同参数数量: {different_count} ({different_count/total_params*100:.2f}%)")
    
    return different_params

def analyze_different_layers(different_params):
    """
    分析不同参数分布在哪些层
    
    参数:
    different_params: 包含不同参数的字典
    
    返回:
    layer_stats: 每层不同参数的统计信息
    """
    layer_stats = {}
    
    for param_name, stats in different_params.items():
        # 提取层名（假设参数名格式为 layer_name.sub_module_name.weight）
        parts = param_name.split('.')
        if len(parts) > 1:
            layer_name = parts[0]
        else:
            layer_name = param_name
        
        if layer_name not in layer_stats:
            layer_stats[layer_name] = {
                'param_count': 0,
                'max_diff': 0,
                'avg_diff': [],
                'params': []
            }
        
        layer_stats[layer_name]['param_count'] += 1
        layer_stats[layer_name]['max_diff'] = max(layer_stats[layer_name]['max_diff'], stats['max_diff'])
        layer_stats[layer_name]['avg_diff'].append(stats['mean_diff'])
        layer_stats[layer_name]['params'].append(param_name)
    
    # 计算平均差异
    for layer in layer_stats.values():
        layer['avg_diff'] = sum(layer['avg_diff']) / len(layer['avg_diff'])
    
    return layer_stats

# 执行参数比较
print("开始比较两个模型的参数...")
different_params = compare_model_parameters(model1, model2)

# 输出差异最大的前10个参数
if different_params:
    print("\n差异最大的前10个参数:")
    sorted_params = sorted(different_params.items(), 
                          key=lambda x: x[1]['max_diff'], 
                          reverse=True)
    for i, (name, stats) in enumerate(sorted_params[:10], 1):
        print(f"{i}. 参数: {name}")
        print(f"   最大差异: {stats['max_diff']:.8f}")
        print(f"   平均差异: {stats['mean_diff']:.8f}")
        print(f"   形状: {stats['shape']}")
        print(f"   model1范围: [{stats['param1_min']:.6f}, {stats['param1_max']:.6f}]")
        print(f"   model2范围: [{stats['param2_min']:.6f}, {stats['param2_max']:.6f}]")
        print()
    
    # 分析不同层的分布
    layer_stats = analyze_different_layers(different_params)
    print("\n不同参数在各层的分布:")
    sorted_layers = sorted(layer_stats.items(), 
                          key=lambda x: x[1]['param_count'], 
                          reverse=True)
    for i, (layer, stats) in enumerate(sorted_layers[:10], 1):
        print(f"{i}. 层: {layer}")
        print(f"   不同参数数量: {stats['param_count']}")
        print(f"   最大差异: {stats['max_diff']:.8f}")
        print(f"   平均差异: {stats['avg_diff']:.8f}")
        print()
    
    # 保存结果到文件
    import json
    with open('model_comparison_results.json', 'w') as f:
        json.dump(different_params, f, indent=4)
    print("完整比较结果已保存至 'model_comparison_results.json'")
else:
    print("两个模型的参数完全相同!")
