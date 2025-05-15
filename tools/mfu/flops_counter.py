from functools import lru_cache
from recovlm.utils.ds_utils import format_dict_or_list
import platform
import subprocess
import os
import re


def s(x):
    if isinstance(x, list): return sum(x)
    else: return x

@lru_cache(maxsize=1)
def get_gpu_model():
    """
    获取当前系统中NVIDIA显卡的型号信息
    
    返回:
    str: 显卡型号名称，如果无法检测则返回 "Unknown"
    """
    try:
        # 优先尝试使用nvidia-smi（最可靠的方法）
        if platform.system() in ["Linux", "Darwin"]:  # Linux/macOS
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return result.stdout.strip()
        
        elif platform.system() == "Windows":  # Windows
            # 尝试使用nvidia-smi（如果在PATH中）
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                shell=True
            )
            if result.returncode == 0:
                return result.stdout.strip()
            
            # 备选方案：使用Windows Management Instrumentation (WMI)
            try:
                import wmi
                c = wmi.WMI()
                gpus = c.Win32_VideoController()
                for gpu in gpus:
                    if "NVIDIA" in gpu.Name:
                        return gpu.Name
            except ImportError:
                pass
    
        # 备选方案：检查CUDA库（需要PyTorch或TensorFlow）
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.get_device_name(0)
        except ImportError:
            pass
            
        try:
            import tensorflow as tf
            if tf.test.is_gpu_available():
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    details = tf.config.experimental.get_device_details(gpus[0])
                    return details.get('device_name', 'NVIDIA GPU')
        except ImportError:
            pass
    
        # 最后手段：检查系统环境变量或驱动文件
        if platform.system() == "Linux":
            # 检查驱动文件
            if os.path.exists("/proc/driver/nvidia/version"):
                with open("/proc/driver/nvidia/version", "r") as f:
                    first_line = f.readline().strip()
                    match = re.search(r"NVIDIA driver \S+ for (\S+)", first_line)
                    if match:
                        return match.group(1)
    
    except Exception as e:
        print(f"检测显卡型号时出错: {e}")
    
    return "Unknown"


@lru_cache(maxsize=1)
def is_h800():
    gpu_model = get_gpu_model()
    return gpu_model.split('\n')[0].strip()=='NVIDIA H800'

@lru_cache(maxsize=1)
def gpu_flops():
    if is_h800():
        return 989e12
    else:
        return 312e12

def calculate_decoder_flops_v1(num_head, head_dim, hidden_size, intermediate_size, kv_heads=None, is_causal=False, seq_len=1, batch_size=1, linear_factor=2, ffn_layers=2):
    """
    计算Transformer解码器层的FLOPs
    
    参数:
    num_head (int): 注意力头的数量
    head_dim (int): 每个注意力头的维度
    hidden_size (int): 隐藏层大小
    intermediate_size (int): FFN中间层大小
    kv_heads (int, optional): KV注意力头的数量，用于Group Attention，默认None表示不使用
    is_causal (bool): 是否使用因果掩码
    seq_len (int): 输入序列长度
    batch_size (int): 批处理大小，将序列分割为多个样本
    
    返回:
    dict: 包含各步骤FLOPs和总FLOPs的字典
    """
    # 默认KV头数量等于查询头数量
    if kv_heads is None:
        kv_heads = num_head

    # 计算每个样本的序列长度
    seq_len_per_sample = None if isinstance(seq_len, list) else seq_len // batch_size
    
    s_seq_len = s(seq_len)

    # 注意力计算FLOPs
    # QKV投影 (不受batch_size影响)
    q_flops = linear_factor * s_seq_len * hidden_size * (num_head * head_dim)
    k_flops = linear_factor * s_seq_len * hidden_size * (kv_heads * head_dim)
    v_flops = linear_factor * s_seq_len * hidden_size * (kv_heads * head_dim)

    # 注意力分数计算 [seq_len_per_sample, num_head, seq_len_per_sample, head_dim]
    if isinstance(seq_len, list):
        attn_scores_flops = 0
        for i, seq_len_per_sample in enumerate(seq_len):
            attn_scores_flops += linear_factor * num_head * seq_len_per_sample * seq_len_per_sample * head_dim
    else:
        attn_scores_flops = linear_factor * num_head * seq_len_per_sample * seq_len_per_sample * head_dim * batch_size


    # 因果掩码（如果启用）会减少一半的注意力计算
    if is_causal:
        # 因果掩码下三角区域计算量: n(n+1)/2 ≈ n²/2
        attn_scores_flops *= 0.5
    
    attn_v_flops = attn_scores_flops

    # 注意力输出投影 (不受batch_size影响)
    attn_out_flops = linear_factor * s_seq_len * (num_head * head_dim) * hidden_size * ffn_layers
    
    # 注意力总FLOPs
    attention_flops = q_flops + k_flops + v_flops + attn_scores_flops + attn_v_flops + attn_out_flops
    
    # FFN层FLOPs (不受batch_size影响)
    ffn_1_flops = linear_factor * s_seq_len * hidden_size * intermediate_size
    ffn_2_flops = linear_factor * s_seq_len * intermediate_size * hidden_size
    ffn_flops = ffn_1_flops + ffn_2_flops

    # 总FLOPs
    total_flops = attention_flops + ffn_flops

    return {
        'total_flops': total_flops,
        'attention': {
            'q_proj': q_flops,
            'k_proj': k_flops,
            'v_proj': v_flops,
            'attn_scores': attn_scores_flops,
            'attn_v': attn_v_flops,
            'attn_out': attn_out_flops,
            'total': attention_flops
        },
        'ffn': {
            'fc1': ffn_1_flops,
            'fc2': ffn_2_flops,
            'total': ffn_flops
        },
        'batch_info': {
            'batch_size': batch_size,
            'seq_len_per_sample': seq_len_per_sample
        }
    }

import easydict




def calculate_decoder_layers_flops(num_head, head_dim, hidden_size, intermediate_size,
                                 kv_heads=None, is_causal=False, seq_len=1, num_layers=1,
                                 linear_factor=2, batch_size=1, ffn_layers=2):
    """
    计算多层Transformer解码器的FLOPs
    
    参数:
    num_head (int): 注意力头的数量
    head_dim (int): 每个注意力头的维度
    hidden_size (int): 隐藏层大小
    intermediate_size (int): FFN中间层大小
    kv_heads (int, optional): KV注意力头的数量，用于Group Attention
    is_causal (bool): 是否使用因果掩码
    seq_len (int): 输入序列长度
    num_layers (int): 解码器层数
    backend (function): 使用的计算方法，默认为v1
    linear_factor (int): 线性计算因子，用于调整计算量
    
    返回:
    dict: 包含各层详细FLOPs和总FLOPs的字典
    """
    layers_flops = []
    total_flops = 0
    # 计算每一层的FLOPs
    for layer_idx in range(num_layers):
        layer_flops = calculate_decoder_flops_v1(
            num_head=num_head,
            head_dim=head_dim,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            kv_heads=kv_heads,
            is_causal=is_causal,
            seq_len=seq_len,
            linear_factor=linear_factor,
            batch_size=batch_size,
            ffn_layers=ffn_layers
        )
        layers_flops.append({
            'layer_index': layer_idx,
            **layer_flops
        })
        total_flops += layer_flops['total_flops']

    # 构建返回字典
    return {
        'total_flops': total_flops,
        'per_layer_flops': layers_flops[0],
        'avg_flops_per_layer': total_flops / num_layers if num_layers > 0 else 0,
        'num_layers': num_layers,
        'backend': str(calculate_decoder_flops_v1),
    }


def calculate_vlm_flops(vit_params, llm_params, linear_factor=2):
    """
    计算VLM(Vision-Language Model)的总计算量
    
    参数:
    vit_params (easydict): 包含ViT参数的对象，需要包含以下字段:
        num_head: ViT的注意力头数量
        head_dim: ViT每个注意力头的维度
        hidden_size: ViT的隐藏层大小
        intermediate_size: ViT的FFN中间层大小
        num_layers: ViT的层数
        kv_heads: ViT的KV注意力头数量(可选)
        seq_len: ViT的序列长度(通常为patch数量+1)
        batch_size: ViT的批处理大小
    
    llm_params (easydict): 包含LLM参数的对象，需要包含以下字段:
        num_head: LLM的注意力头数量
        head_dim: LLM每个注意力头的维度
        hidden_size: LLM的隐藏层大小
        intermediate_size: LLM的FFN中间层大小
        num_layers: LLM的层数
        kv_heads: LLM的KV注意力头数量(可选)
        is_causal: LLM是否使用因果注意力(默认为True)
        seq_len: LLM的序列长度
        batch_size: LLM的批处理大小
    
    通用参数:
        backend: 使用的计算方法(默认为v1)
        linear_factor: 线性计算因子
    
    返回:
    dict: 包含ViT、LLM详细计算量和总计算量的字典
    """
    # 计算ViT的计算量
    vit_flops = calculate_decoder_layers_flops(
        num_head=vit_params.num_head,
        head_dim=vit_params.head_dim,
        hidden_size=vit_params.hidden_size,
        intermediate_size=vit_params.intermediate_size,
        num_layers=vit_params.num_layers,
        kv_heads=vit_params.get('kv_heads', None),
        is_causal=False,  # ViT通常不使用因果注意力
        seq_len=vit_params.seq_len,
        batch_size=vit_params.get('batch_size', 1),
        linear_factor=linear_factor,
        ffn_layers=2
    )

    vit2llm_flops = linear_factor * s(vit_params.seq_len) * (vit_params.hidden_size * llm_params.hidden_size + llm_params.hidden_size * llm_params.hidden_size)
    vit_flops['total_flops'] += vit2llm_flops
    vit_flops['vit2llm_flops'] = vit2llm_flops
    
    # 计算LLM的计算量
    llm_flops = calculate_decoder_layers_flops(
        num_head=llm_params.num_head,
        head_dim=llm_params.head_dim,
        hidden_size=llm_params.hidden_size,
        intermediate_size=llm_params.intermediate_size,
        num_layers=llm_params.num_layers,
        kv_heads=llm_params.get('kv_heads', None),
        is_causal=llm_params.get('is_causal', True),
        seq_len=llm_params.seq_len,
        batch_size=llm_params.get('batch_size', 1),
        linear_factor=linear_factor,
        ffn_layers=3
    )
    
    lm_head_flops = linear_factor * s(llm_params.seq_len) * (llm_params.hidden_size * llm_params.vocab_size)
    llm_flops['total_flops'] += lm_head_flops
    llm_flops['lm_head_flops'] = lm_head_flops
    
    # 计算总FLOPs
    total_flops = vit_flops['total_flops'] + llm_flops['total_flops']
    _gpu_flops = gpu_flops()
    return {
        'total_flops': total_flops,
        'vit': vit_flops,
        'llm': llm_flops,
        'vit_percentage': vit_flops['total_flops'] / total_flops * 100 if total_flops > 0 else 0,
        'llm_percentage': llm_flops['total_flops'] / total_flops * 100 if total_flops > 0 else 0,
        'vit_total_flops*3(T)': vit_flops['total_flops'] * 3 / 1e12,
        'llm_total_flops*3(T)': llm_flops['total_flops'] * 3 / 1e12,
        'total_flops*3(T)': total_flops * 3 / 1e12,
        'total_flops/gpu_flops': total_flops * 3 / _gpu_flops,
        'gpu_flops': _gpu_flops
    }


@lru_cache(maxsize=32)
def extract_model_params(config_path):
    """
    从模型配置JSON文件中提取Transformer和Vision模块的参数
    支持Qwen3和InternVL架构
    
    参数:
    config_path (str): JSON配置文件路径
    
    返回:
    tuple: 包含两个字典的元组
        - transformer_params: Transformer模块参数
        - vision_params: Vision模块参数
    """
    import json

    # 读取JSON配置文件
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 判断模型架构类型
    if 'architectures' in config and 'InternVLChatModel' in config['architectures']:
        # InternVL架构处理逻辑
        llm_config = config['llm_config']

        # 提取LLM参数
        transformer_params = {
            'num_head': llm_config['num_attention_heads'],
            'head_dim': llm_config['hidden_size'] // llm_config['num_attention_heads'],
            'hidden_size': llm_config['hidden_size'],
            'intermediate_size': llm_config['intermediate_size'],
            'kv_heads': llm_config.get('num_key_value_heads', llm_config['num_attention_heads']),
            'num_layers': llm_config['num_hidden_layers'],
            'vocab_size': config['vocab_size']
        }
        
        vision_config = config['vision_config']
        # 提取Vision参数 
        vision_params = {
            'num_head': vision_config['num_heads'],
            'head_dim': vision_config['hidden_size'] / vision_config['num_heads'],
            'hidden_size': vision_config['hidden_size'],
            'intermediate_size': config['intermediate_size'],
            'num_layers': vision_config['depth'],
        }
        vision_params = {k: v for k, v in vision_params.items() if v is not None}
    elif 'architectures' in config and 'Qwen2_5_VLForConditionalGeneration' in config['architectures']:
        # Qwen2.5 VL架构处理逻辑
        transformer_params = {
            'num_head': config['num_attention_heads'],
            'head_dim': config['hidden_size'] / config['num_attention_heads'],
            'hidden_size': config['hidden_size'],
            'intermediate_size': config['intermediate_size'],
            'kv_heads': config['num_key_value_heads'],
            'num_layers': config['num_hidden_layers'],
            'vocab_size': config['vocab_size']
        }
        vision_config = config['vision_config']
        vision_params = {
            'num_head': vision_config['num_heads'],
            'head_dim': vision_config['hidden_size'] / vision_config['num_heads'],
            'hidden_size': vision_config['hidden_size'],
            'intermediate_size': vision_config['intermediate_size'],
            'num_layers': vision_config['depth'],
            
        }
    else:
        # Qwen3架构处理逻辑 (保持原有逻辑)
        transformer_params = {
            'num_head': config['num_attention_heads'],
            'head_dim': config['head_dim'],
            'hidden_size': config['hidden_size'],
            'intermediate_size': config['intermediate_size'],
            'kv_heads': config['num_key_value_heads'],
            'num_layers': config['num_hidden_layers'],
            'vocab_size': config['vocab_size']
        }
        vision_config = config['vision_config']
        vision_params = {
            'num_head':vision_config['num_heads'],
            'head_dim': vision_config['hidden_size'] / vision_config['num_heads'],
            'hidden_size': vision_config['hidden_size'],
            'intermediate_size': vision_config['intermediate_size'],
            'num_layers': vision_config['depth'],
        }
        vision_params = {k: v for k, v in vision_params.items() if v is not None}
    
    return transformer_params, vision_params
    


def calc_mfu(config_path, total_seq_len, image_token_merged_len, llm_batch_size, image_batch_size=None, secs_per_step=None):
    if image_batch_size is None: image_batch_size = llm_batch_size
    transformer_params, vision_params = extract_model_params(
        config_path
    )
    llm_params = easydict.EasyDict({
        **transformer_params,
        'is_causal': False,
        'seq_len': total_seq_len, 
        'batch_size': llm_batch_size
    })

    vit_params = easydict.EasyDict({
        **vision_params,
        'is_causal': False,
        'seq_len': [x*4 for x in image_token_merged_len] if isinstance(image_token_merged_len, list) else image_token_merged_len * 4, 
        'batch_size': image_batch_size
    })

    flops = calculate_vlm_flops(vit_params, llm_params)

    flops['input_args'] = easydict.EasyDict(
        config_path=config_path,
        total_seq_len=total_seq_len,
        image_token_merged_len=image_token_merged_len,
        llm_batch_size=llm_batch_size,
        image_batch_size=image_batch_size,
        secs_per_step=secs_per_step
    )
    if secs_per_step is not None:
        flops['mfu'] = flops['total_flops/gpu_flops'] / secs_per_step
    return flops

if 0:
    config_path = '/Users/lingzhixin/Desktop/work/LLMreco/grpo_rlmain/recovlm0515/recovlm/tools/mfu/qwen3_1.7b_navit.json'
    transformer_params, vision_params = extract_model_params(
        config_path
    )
    # 转换为calculate_vlm_flops需要的格式
    llm_params = easydict.EasyDict({
        **transformer_params,
        'is_causal': False,
        'seq_len': 23000, 
        'batch_size': 63
    })

    vit_params = easydict.EasyDict({
        **vision_params,
        'is_causal': False,
        'seq_len': 20239 * 4, 
        'batch_size': 63
    })

    # 计算FLOPs
    flops = calculate_vlm_flops(vit_params, llm_params)

    print('config 路径:')
    print(config_path)

    print('VLM FLOPs llm计算参数:')
    print(format_dict_or_list(llm_params))

    print('VLM FLOPs vit计算参数:')
    print(format_dict_or_list(vit_params))


    print('config 路径:')
    print(config_path)

    # 打印结果
    print('VLM FLOPs计算结果:')
    print(format_dict_or_list(flops))


    print(f"=" * 40)

if __name__=='__main__':
    mfu = calc_mfu(
        '/Users/lingzhixin/Desktop/work/LLMreco/grpo_rlmain/recovlm0515/recovlm/tools/mfu/qwen3_1.7b_navitd.json',
        total_seq_len=4800*2,
        image_token_merged_len=4800,
        llm_batch_size=48,
        secs_per_step=10
    )

    # 打印结果
    print('VLM FLOPs计算结果:')
    print(format_dict_or_list(mfu))

    mfu = calc_mfu(
        '/Users/lingzhixin/Desktop/work/LLMreco/grpo_rlmain/recovlm0515/recovlm/tools/mfu/qwen3_1.7b_navitd.json',
        total_seq_len=[4800*2 // 48 for _ in range(48)],
        image_token_merged_len=[4800 // 48 for _ in range(48)],
        llm_batch_size=48,
        secs_per_step=10
    )

    # 打印结果
    print('VLM FLOPs计算结果:')
    print(format_dict_or_list(mfu))

'''
2B 模型
样本吞吐10, image token 1324, 132*4 token

105

10.6%




8B模型
240卡
llm_batchsize=48
llm_seq_len=17496
vit_token_len=15315
mfu = 9%
  "vit_total_flops*3(T)": 237.74510186496,
  "llm_total_flops*3(T)": 613.585621352448

2B模型
368卡
llm_batchsize=63
llm_seq_len=23000
vit_token_len=20239
mfu= 3%

  "llm_percentage": 34.79381463257135,
  "vit_total_flops*3(T)": 314.5511215104,
  "llm_total_flops*3(T)": 167.8434852864
'''
