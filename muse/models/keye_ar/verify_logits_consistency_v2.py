"""
验证KeyeARModel和KeyeForConditionalGeneration前向logits一致性的脚本
基于test_forward_v2的实现，使用load_keye_ar_config函数初始化KeyeARModel
"""

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["nosp"] = 'true'

import json
import torch
import numpy as np
import warnings
from pathlib import Path
from PIL import Image, ImageDraw
from transformers import AutoProcessor
from contextlib import nullcontext
from collections import defaultdict

# 导入模型相关模块
from muse.models.keye_ar.ar_ori import KeyeForConditionalGeneration
from muse.models.keye_ar.modeling import KeyeARModel
from muse.models.keye_ar.keye_vl_utils import process_vision_info
from muse.config import KeyeARConfig, UnifiedQwen3Config, KeyeTokenizerConfig, UnifiedTokenDecoderConfig, KeyeVisionConfig


def get_config_value(config_dict, key, default_value, config_name=""):
    """从配置字典中获取值，如果不存在则使用默认值并发出警告"""
    value = config_dict.get(key)
    if value is None:
        value = default_value
        config_source = f" in {config_name}" if config_name else ""
        warnings.warn(f"{key} not found{config_source}, using default value: {default_value}")
    return value

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")




def load_keye_ar_config(conf_path):
    """直接从conf.json加载KeyeARConfig（从test_modeling_forward.py复制）"""
    with open(conf_path, 'r') as f:
        conf_data = json.load(f)
    
    # 构造KeyeVisionConfig
    vision_config_data = conf_data.get('vision_config', {})
    vision_config_inner = vision_config_data.get('vision_config', {})
    
    # 获取vision_config_inner中的字段
    image_size = get_config_value(vision_config_inner, 'image_size', 384, "vision_config")
    patch_size = get_config_value(vision_config_inner, 'patch_size', 14, "vision_config")
    hidden_size = get_config_value(vision_config_inner, 'hidden_size', 1152, "vision_config")
    num_hidden_layers = get_config_value(vision_config_inner, 'num_hidden_layers', 27, "vision_config")
    num_attention_heads = get_config_value(vision_config_inner, 'num_attention_heads', 16, "vision_config")
    intermediate_size = get_config_value(vision_config_inner, 'intermediate_size', 4304, "vision_config")
    hidden_act = get_config_value(vision_config_inner, 'hidden_act', 'gelu_pytorch_tanh', "vision_config")
    layer_norm_eps = get_config_value(vision_config_inner, 'layer_norm_eps', 1e-6, "vision_config")
    attention_dropout = get_config_value(vision_config_inner, 'attention_dropout', 0.0, "vision_config")
    rope_theta = get_config_value(vision_config_inner, 'rope_theta', 10000.0, "vision_config")
    
    keye_vision_config = KeyeVisionConfig(
        model_class="KeyeVL1_5VisionModel",  # 添加model_class字段
        image_size=image_size,
        patch_size=patch_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        hidden_act=hidden_act,
        layer_norm_eps=layer_norm_eps,
        attention_dropout=attention_dropout,
        rope_theta=rope_theta,
    )
    
    # 构造KeyeTokenizerConfig
    codebook_size = get_config_value(vision_config_data, 'codebook_size', 65536, "vision_config_data")
    embedding_dim = get_config_value(vision_config_data, 'embedding_dim', 128, "vision_config_data")
    init_embedding_dim = get_config_value(vision_config_data, 'init_embedding_dim', 4096, "vision_config_data")
    llm_hidden_size = get_config_value(vision_config_data, 'llm_hidden_size', 4096, "vision_config_data")
    n_q_tokens = get_config_value(vision_config_data, 'n_q_tokens', 8, "vision_config_data")
    split_voc = get_config_value(vision_config_data, 'split_voc', 1, "vision_config_data")

    keye_tokenizer_config = KeyeTokenizerConfig(
        model_class="KeyeImageTokenizer",  # 添加model_class字段
        vision_config=keye_vision_config,
        codebook_size=codebook_size,
        embedding_dim=embedding_dim,
        init_embedding_dim=init_embedding_dim,
        llm_hidden_size=llm_hidden_size,
        n_q_tokens=n_q_tokens,
        split_voc=split_voc
    )
    
    print(f"keye_tokenizer_config={keye_tokenizer_config}")
    # 构造UnifiedTokenDecoderConfig
    token_head_dim = get_config_value(conf_data, 'token_head_dim', 512, "conf_data")
    token_head_nhead = get_config_value(conf_data, 'token_head_nhead', 4, "conf_data")
    token_head_intermediate_dim = get_config_value(conf_data, 'token_head_intermediate_dim', 2048, "conf_data")
    token_head_num_layers = get_config_value(conf_data, 'token_head_num_layers', 1, "conf_data")
    hidden_size = get_config_value(conf_data, 'hidden_size', 4096, "conf_data")

    unified_token_decoder_config = UnifiedTokenDecoderConfig(
        model_class="UnifiedTokenDecoder",  # 添加model_class字段
        vocab_size=codebook_size,
        d_model=token_head_dim,
        nhead=token_head_nhead,
        num_layers=1,  # 默认值
        dim_feedforward=token_head_intermediate_dim,
        input_dim=hidden_size,
        reduce=True
    )
    
    # 构造UnifiedQwen3Config
    vocab_size = get_config_value(conf_data, 'vocab_size', 151936, "conf_data")
    
    num_hidden_layers = get_config_value(conf_data, 'num_hidden_layers', 36, "conf_data")
    num_attention_heads = get_config_value(conf_data, 'num_attention_heads', 32, "conf_data")
    num_key_value_heads = get_config_value(conf_data, 'num_key_value_heads', 8, "conf_data")
    head_dim = get_config_value(conf_data, 'head_dim', 128, "conf_data")
    intermediate_size = get_config_value(conf_data, 'intermediate_size', 12288, "conf_data")
    hidden_act = get_config_value(conf_data, 'hidden_act', 'silu', "conf_data")
    rms_norm_eps = get_config_value(conf_data, 'rms_norm_eps', 1e-6, "conf_data")
    attention_dropout = get_config_value(conf_data, 'attention_dropout', 0.0, "conf_data")
    rope_theta = get_config_value(conf_data, 'rope_theta', 1000000, "conf_data")
    max_position_embeddings = get_config_value(conf_data, 'max_position_embeddings', 40960, "conf_data")
    tie_word_embeddings = get_config_value(conf_data, 'tie_word_embeddings', False, "conf_data")

    image_token_id = conf_data.get('image_token_id')
    pad_token_id = conf_data.get('pad_token_id')
    q_eos_token = conf_data.get('q_eos_token')
    
    unified_qwen_config = UnifiedQwen3Config(
        model_class="Qwen3Model",  # 添加model_class字段
        vocab_size=vocab_size,
        embed_dim=hidden_size,
        num_layers=num_hidden_layers,
        num_heads=num_attention_heads,
        num_kv_heads=num_key_value_heads,
        head_dim=head_dim,
        intermediate_dim=intermediate_size,
        hidden_act=hidden_act,
        norm_eps=rms_norm_eps,
        attn_dropout=attention_dropout,
        tie_word_embeddings=tie_word_embeddings,
        rope_theta=rope_theta,
        max_seq_len=max_position_embeddings,
image_token_id=image_token_id,
        pad_token_id=pad_token_id,
q_eos_token=q_eos_token,
        codebook_size=codebook_size,
        n_q_tokens=n_q_tokens,
        token_head_d_model=token_head_dim,
        token_head_nheads=token_head_nhead,
        token_head_dim_feedforward=token_head_intermediate_dim,
        token_head_num_layers=token_head_num_layers,
    )

    # 构造KeyeARConfig
    keye_ar_config = KeyeARConfig(
        model_class="KeyeARModel",  # 添加model_class字段
        qwen_config=unified_qwen_config,
        tokenizer_config=keye_tokenizer_config,
        token_decoder_config=unified_token_decoder_config,
    )
    
    return keye_ar_config


def generate_circle_image(size=(100, 100), fill_color=(0, 0, 0), outline_color=(255, 255, 255), outline_width=5):
    """生成测试用的圆形图像"""
    image = Image.new('RGB', size, color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    x_center, y_center = size[0] // 2, size[1] // 2
    radius = min(size[0], size[1]) // 2
    draw.ellipse([x_center - radius, y_center - radius, x_center + radius, y_center + radius],
                 fill=fill_color, outline=outline_color, width=outline_width)
    return image

def process_im_message(processor, image):
    messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
            ],
        }]
    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True  # 开启生成提示
    )

    print(f"text={text}")

    image_inputs, video_inputs = process_vision_info(messages)

    # 构建原始输入（纯有效Token，无任何Pad）
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=False,  # 强制关闭Pad，确保原始输入无多余Token
        truncation=False,
        return_tensors="pt",
    ).to(device)
    return inputs

def load_keye_for_conditional_generation(output_model_dir, device):
    """加载KeyeForConditionalGeneration模型（ground truth）"""
    model = KeyeForConditionalGeneration.from_pretrained(
        output_model_dir, 
        _attn_implementation="flash_attention_2", 
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True
    )
    model.config.output_one_token = model.output_one_token = False
    model.token_head.use_flash_attn = True
    model = model.to(device).bfloat16()
    return model


def load_keye_ar_model_v2(output_model_dir, device):
    """加载KeyeARModel模型（基于test_forward_v2的实现）"""
    # 加载processor和配置
    processor = AutoProcessor.from_pretrained(output_model_dir, trust_remote_code=True)
    
    # 直接从conf.json加载KeyeARConfig（使用load_keye_ar_config函数）
    config = load_keye_ar_config("muse/models/keye_ar/conf.json")
    
    # 创建模型实例
    model = KeyeARModel(config)
    
    # 从KeyeForConditionalGeneration获取state_dict并转换
    keye_model = KeyeForConditionalGeneration.from_pretrained(
        output_model_dir, 
        _attn_implementation="flash_attention_2", 
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True
    )
    
    # 获取KeyeForConditionalGeneration的state_dict
    keye_state_dict = keye_model.state_dict()
    
    # 转换为KeyeARModel的state_dict
    converted_state_dict = model.convert_hf_state_dict(keye_state_dict, tie_word_embeddings=False)
    model.load_state_dict(converted_state_dict, strict=True)
    
    # 将模型移到设备并转换为bfloat16精度
    model = model.to(device).bfloat16()
    
    return model, processor


def process_message(messages, processor, device, add_generation_prompt=True, padding=False):
    """处理消息，生成模型输入"""
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=padding,
        truncation=False,
        return_tensors="pt",
    ).to(device)
    
    # 转换到bfloat16精度
    def _cast_inputs_to_bf16(batch):
        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                batch[k] = v.to(dtype=torch.bfloat16)
        return batch
    
    inputs = _cast_inputs_to_bf16(inputs)
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}


class LayerAlignmentHook:
    """为模型层添加forward hook来对齐输出"""
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.layer_outputs = defaultdict(list)
        self.hooks = []
        
    def register_hooks(self, model):
        """为模型的关键层注册forward hook"""
        print(f"为{self.model_name}注册层对齐hook...")
        
        # 检测模型类型并应用相应的hook注册逻辑
        if hasattr(model, 'model') and hasattr(model.model, 'model'):
            # KeyeARModel结构：model -> model.model -> UnifiedTransformerDecoder
            self._register_keye_ar_hooks(model)
        else:
            # KeyeForConditionalGeneration结构
            self._register_keye_conditional_generation_hooks(model)
    
    def _register_keye_conditional_generation_hooks(self, model):
        """为KeyeForConditionalGeneration注册hook"""
        # 为Qwen3Model的transformer层注册hook
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            for i, layer in enumerate(model.model.layers):
                hook = self._create_layer_hook(f"transformer_layer_{i}")
                self.hooks.append(layer.register_forward_hook(hook))
                print(f"  注册transformer层 {i}")
        
        # 为embedding层注册hook
        if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            hook = self._create_layer_hook("embedding")
            self.hooks.append(model.model.embed_tokens.register_forward_hook(hook))
            print(f"  注册embedding层")
        
        # 为norm层注册hook
        if hasattr(model, 'model') and hasattr(model.model, 'norm'):
            hook = self._create_layer_hook("final_norm")
            self.hooks.append(model.model.norm.register_forward_hook(hook))
            print(f"  注册final_norm层")
        
        # 为lm_head注册hook
        if hasattr(model, 'lm_head'):
            hook = self._create_layer_hook("lm_head")
            self.hooks.append(model.lm_head.register_forward_hook(hook))
            print(f"  注册lm_head层")
        
        # 为token_head注册hook（KeyeForConditionalGeneration特有）
        if hasattr(model, 'token_head'):
            hook = self._create_layer_hook("token_head")
            self.hooks.append(model.token_head.register_forward_hook(hook))
            print(f"  注册token_head层")
            
            # 为token_head内部的transformer层注册hook
            if hasattr(model.token_head, 'transformer') and hasattr(model.token_head.transformer, 'layers'):
                for i, layer in enumerate(model.token_head.transformer.layers):
                    hook = self._create_layer_hook(f"token_head_transformer_layer_{i}")
                    self.hooks.append(layer.register_forward_hook(hook))
                    print(f"  注册token_head transformer层 {i}")
    
    def _register_keye_ar_hooks(self, model):
        """为KeyeARModel注册hook"""
        # KeyeARModel结构：model -> model.model -> UnifiedTransformerDecoder
        
        # 为UnifiedTransformerDecoder的transformer层注册hook
        if hasattr(model.model, 'model') and hasattr(model.model.model, 'layers'):
            for i, layer in enumerate(model.model.model.layers):
                hook = self._create_layer_hook(f"transformer_layer_{i}")
                self.hooks.append(layer.register_forward_hook(hook))
                print(f"  注册transformer层 {i}")
        
        # 为embedding层注册hook
        if hasattr(model.model, 'model') and hasattr(model.model.model, 'tok_embeddings'):
            hook = self._create_layer_hook("embedding")
            self.hooks.append(model.model.model.tok_embeddings.register_forward_hook(hook))
            print(f"  注册embedding层")
        
        # 为norm层注册hook
        if hasattr(model.model, 'model') and hasattr(model.model.model, 'norm'):
            hook = self._create_layer_hook("final_norm")
            self.hooks.append(model.model.model.norm.register_forward_hook(hook))
            print(f"  注册final_norm层")
        
        # 为lm_head注册hook
        if hasattr(model, 'lm_head'):
            hook = self._create_layer_hook("lm_head")
            self.hooks.append(model.lm_head.register_forward_hook(hook))
            print(f"  注册lm_head层")
        
        # 为token_head注册hook（KeyeARModel特有）
        if hasattr(model.model, 'model') and hasattr(model.model.model, 'token_head'):
            hook = self._create_layer_hook("token_head")
            self.hooks.append(model.model.model.token_head.register_forward_hook(hook))
            print(f"  注册token_head层")
            
            # 为token_head内部的transformer层注册hook
            if hasattr(model.model.model.token_head, 'transformer') and hasattr(model.model.model.token_head.transformer, 'layers'):
                for i, layer in enumerate(model.model.model.token_head.transformer.layers):
                    hook = self._create_layer_hook(f"token_head_transformer_layer_{i}")
                    self.hooks.append(layer.register_forward_hook(hook))
                    print(f"  注册token_head transformer层 {i}")
    
    def _create_layer_hook(self, layer_name):
        """创建forward hook函数"""
        def hook(module, input, output):
            if len(self.layer_outputs[layer_name]) == 1:
                print(f"跳过{layer_name}的第二个输出")
                return
            
            # 存储层的输出
            if isinstance(output, tuple):
                # 对于返回tuple的层，取第一个元素（通常是hidden states）
                self.layer_outputs[layer_name].append(output[0].detach().cpu())
            else:
                self.layer_outputs[layer_name].append(output.detach().cpu())
        return hook
    
    def remove_hooks(self):
        """移除所有hook"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def clear_outputs(self):
        """清空存储的输出"""
        self.layer_outputs.clear

def compare_layer_outputs(hook1, hook2, tolerance=1e-5):
    """比较两个hook记录的层输出"""
    print("\n" + "="*80)
    print("层对齐分析报告")
    print("="*80)
    
    all_success = True
    layer_comparisons = {}
    
    # 获取两个hook中共同的层
    common_layers = set(hook1.layer_outputs.keys()) & set(hook2.layer_outputs.keys())
    unique_layers1 = set(hook1.layer_outputs.keys()) - common_layers
    unique_layers2 = set(hook2.layer_outputs.keys()) - common_layers
    
    if unique_layers1:
        print(f"⚠️ {hook1.model_name} 独有的层: {sorted(unique_layers1)}")
    if unique_layers2:
        print(f"⚠️ {hook2.model_name} 独有的层: {sorted(unique_layers2)}")
    
    for layer_name in sorted(common_layers):
        outputs1 = hook1.layer_outputs[layer_name]
        outputs2 = hook2.layer_outputs[layer_name]
        
        if len(outputs1) != len(outputs2):
            print(f"❌ {layer_name}: 输出数量不匹配 ({len(outputs1)} vs {len(outputs2)})")
            print(f"outputs1({[x.shape for x in outputs1]})={outputs1}")
            print(f"outputs2({[x.shape for x in outputs2]})={outputs2}")
            all_success = False
            layer_comparisons[layer_name] = False
            continue
        
        layer_success = True
        for i, (out1, out2) in enumerate(zip(outputs1, outputs2)):
            # 检查形状是否一致
            if out1.shape != out2.shape:
                print(f"❌ {layer_name}[{i}]: 形状不匹配 {out1.shape} vs {out2.shape}")
                print("\n=== 详细输出信息 ===")
                print(f"{hook1.model_name} 输出 (out1):")
                print(f"  形状: {out1.shape}")
                print(f"  数值范围: [{out1.min().item():.6f}, {out1.max().item():.6f}]")
                print(f"  平均值: {out1.mean().item():.6f}")
                print(f"  标准差: {out1.std().item():.6f}")
                print(f"\n{hook2.model_name} 输出 (out2):")
                print(f"  形状: {out2.shape}")
                print(f"  数值范围: [{out2.min().item():.6f}, {out2.max().item():.6f}]")
                print(f"  平均值: {out2.mean().item():.6f}")
                print(f"  标准差: {out2.std().item():.6f}")
                print(f"\n元素总数: {out1.numel()} vs {out2.numel()}")
                
                # 尝试将out1 reshape到out2的形状
                try:
                    # 计算总元素数是否相同
                    if out1.numel() == out2.numel():
                        out1_reshaped = out1.reshape(out2.shape)
                        print(f"     尝试reshape: {out1.shape} -> {out2.shape}")
                        out1 = out1_reshaped
                    else:
                        print(f"     ❌ 元素总数不匹配，无法reshape")
                        layer_success = False
                        all_success = False
                        return False, layer_comparisons
                except Exception as e:
                    print(f"     ❌ reshape失败: {e}")
                    layer_success = False
                    all_success = False
                    return False, layer_comparisons
            
            # 转换为float32进行精确比较
            out1_f32 = out1.float()
            out2_f32 = out2.float()
            
            # 计算绝对误差
            abs_diff = torch.abs(out1_f32 - out2_f32)
            max_abs_diff = torch.max(abs_diff).item()
            mean_abs_diff = torch.mean(abs_diff).item()
            
            # 计算相对误差
            relative_diff = abs_diff / (torch.abs(out2_f32) + 1e-8)
            max_relative_diff = torch.max(relative_diff).item()
            mean_relative_diff = torch.mean(relative_diff).item()
            
            # 检查是否在容差范围内
            if max_abs_diff > tolerance or max_relative_diff > tolerance:
                print(f"out1_f32={out1_f32}")
                print(f"out2_f32={out2_f32}")
                print(f"❌ {layer_name}[{i}]: 输出不一致")
                print("\n=== 详细输出信息 ===")
                print(f"{hook1.model_name} 输出 (out1):")
                print(f"  形状: {out1.shape}")
                print(f"  数值范围: [{out1.min().item():.6f}, {out1.max().item():.6f}]")
                print(f"  平均值: {out1.mean().item():.6f}")
                print(f"  标准差: {out1.std().item():.6f}")
                print(f"\n{hook2.model_name} 输出 (out2):")
                print(f"  形状: {out2.shape}")
                print(f"  数值范围: [{out2.min().item():.6f}, {out2.max().item():.6f}]")
                print(f"  平均值: {out2.mean().item():.6f}")
                print(f"  标准差: {out2.std().item():.6f}")
                
                print(f"\n=== 差异分析 ===")
                print(f"最大绝对误差: {max_abs_diff:.6e}")
                print(f"平均绝对误差: {mean_abs_diff:.6e}")
                print(f"最大相对误差: {max_relative_diff:.6e}")
                print(f"平均相对误差: {mean_relative_diff:.6e}")
                
                # 输出差异最大的位置
                max_diff_indices = torch.argmax(abs_diff.view(-1))
                max_diff_pos = np.unravel_index(max_diff_indices.item(), out1.shape)
                print(f"最大差异位置: {max_diff_pos}")
                print(f"该位置的值: {hook1.model_name}: {out1_f32[max_diff_pos]:.6f}, {hook2.model_name}: {out2_f32[max_diff_pos]:.6f}")
                print(f"绝对差异: {abs_diff[max_diff_pos]:.6e}")
                print(f"相对差异: {relative_diff[max_diff_pos]:.6e}")
                
                layer_success = False
                all_success = False
                return False, layer_comparisons
            else:
                print(f"✅ {layer_name}[{i}]: 输出一致")
                print(f"     最大绝对误差: {max_abs_diff:.6e}")
                print(f"     最大相对误差: {max_relative_diff:.6e}")
        
        layer_comparisons[layer_name] = layer_success
    
    print("\n" + "="*80)
    if all_success:
        print("🎉 所有层输出完全一致！")
    else:
        print("❌ 部分层输出不一致，请检查上述报告")
    
    return all_success, layer_comparisons


def get_keye_conditional_generation_logits(model, inputs, layer_hook=None):
    """获取KeyeForConditionalGeneration的logits（ground truth）"""
    if layer_hook:
        layer_hook.clear_outputs()
    
    if torch.cuda.is_available():
        autocast_cm = torch.cuda.amp.autocast
    else:
        try:
            autocast_cm = torch.cpu.amp.autocast
        except Exception:
            autocast_cm = nullcontext

    inputs["position_ids"] = torch.arange(0, inputs["input_ids"].size(1)).unsqueeze(0).to(inputs["input_ids"].device)

    with autocast_cm(dtype=torch.bfloat16):
        outputs = model(**inputs)
    
    # KeyeForConditionalGeneration直接返回logits
    if hasattr(outputs, 'logits'):
        return outputs.logits
    else:
        # 如果模型输出是tuple，logits通常是第一个元素
        if isinstance(outputs, tuple) and len(outputs) > 0:
            return outputs[0]
        else:
            raise ValueError("无法获取KeyeForConditionalGeneration的logits")

def get_keye_ar_model_logits(model, inputs, layer_hook=None):
    """获取KeyeARModel的logits"""
    if layer_hook:
        layer_hook.clear_outputs()
    
    # 准备KeyeARModel的输入
    inputs_ar = inputs.copy()
    inputs_ar["position_ids"] = torch.arange(0, inputs_ar["input_ids"].size(1)).unsqueeze(0).to(inputs_ar["input_ids"].device)
    inputs_ar["tokens"] = inputs_ar["input_ids"]
    del inputs_ar["input_ids"]
    
    if torch.cuda.is_available():
        autocast_cm = torch.cuda.amp.autocast
    else:
        try:
            autocast_cm = torch.cpu.amp.autocast
        except Exception:
            autocast_cm = nullcontext

    with autocast_cm(dtype=torch.bfloat16):
        outputs = model(**inputs_ar)
    
    return outputs


def compare_logits(logits1, logits2, model1_name, model2_name, tolerance=1e-5):
    """比较两个logits张量是否一致"""
    print(f"\n=== 比较 {model1_name} 和 {model2_name} 的logits ===")
    print(f"Logits1 shape: {logits1.shape}")
    print(f"Logits2 shape: {logits2.shape}")
    
    # 检查形状是否一致
    if logits1.shape != logits2.shape:
        print(f"❌ 形状不一致: {model1_name} {logits1.shape} vs {model2_name} {logits2.shape}")
        return False
    
    # 转换为float32进行精确比较
    logits1_f32 = logits1.float()
    logits2_f32 = logits2.float()
    
    # 计算绝对误差和相对误差
    abs_diff = torch.abs(logits1_f32 - logits2_f32)
    max_abs_diff = torch.max(abs_diff).item()
    mean_abs_diff = torch.mean(abs_diff).item()
    
    # 计算相对误差（避免除以零）
    relative_diff = abs_diff / (torch.abs(logits2_f32) + 1e-8)
    max_relative_diff = torch.max(relative_diff).item()
    mean_relative_diff = torch.mean(relative_diff).item()
    
    print(f"最大绝对误差: {max_abs_diff:.6e}")
    print(f"平均绝对误差: {mean_abs_diff:.6e}")
    print(f"最大相对误差: {max_relative_diff:.6e}")
    print(f"平均相对误差: {mean_relative_diff:.6e}")
    
    # 检查是否在容差范围内
    if max_abs_diff < tolerance and max_relative_diff < tolerance:
        print("✅ Logits完全一致！")
        return True
    else:
        print("❌ Logits不一致！")
        
        # 输出差异最大的位置
        max_diff_indices = torch.argmax(abs_diff.view(-1))
        max_diff_pos = np.unravel_index(max_diff_indices.item(), logits1.shape)
        print(f"最大差异位置: {max_diff_pos}")
        print(f"该位置的值: {model1_name}: {logits1_f32[max_diff_pos]:.6f}, {model2_name}: {logits2_f32[max_diff_pos]:.6f}")
        
        return False


def main():
    """主函数：验证两个模型的logits一致性"""
    # 设置设备
    print(f"使用设备: {device}")
    
    # 模型路径
    output_model_dir = "/mmu_mllm_hdd_2/zhouyang12/output/Keye/vqar_11.7/run_8b_vis_stage3.29_1e-4/step4000/global_step4000/converted"
    
    try:
        # 加载两个模型
        print("正在加载KeyeForConditionalGeneration...")
        keye_conditional_model = load_keye_for_conditional_generation(output_model_dir, device)
        
        print("正在加载KeyeARModel...")
        keye_ar_model, processor = load_keye_ar_model_v2(output_model_dir, device)
        
        if 0:
            tokens_conditional = keye_conditional_model.forward_image_tokens(**process_im_message(processor, generate_circle_image()))
            tokens_ar = keye_ar_model.forward_image_tokens(**process_im_message(processor, generate_circle_image()))
            print(f"tokens_conditional=\n{tokens_conditional}")
            print(f"tokens_ar=\n{tokens_ar}")
            assert torch.all(tokens_conditional == tokens_ar)

        # 创建层对齐hook
        conditional_hook = LayerAlignmentHook("KeyeForConditionalGeneration")
        ar_hook = LayerAlignmentHook("KeyeARModel")
        
        # 为两个模型注册hook
        conditional_hook.register_hooks(keye_conditional_model)
        ar_hook.register_hooks(keye_ar_model)
        
        # 准备测试消息
        COT_SYSTEM_PROMPT = "You are a helpful assistant."
        messages = [
            {"role": "system",
             "content": [
                 {"type": "text", "text": COT_SYSTEM_PROMPT},
             ], },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": generate_circle_image()},
                    {"type": "text", "text": " What's sum of the first 10 positive integers? After necessary analysis, your final output should follow the format: Final Answer: X."},
                ],
            }
        ]
        
        # 处理输入
        print("处理输入消息...")
        inputs = process_message(messages, processor, device)

        print("获取KeyeARModel的logits...")
        keye_ar_logits = get_keye_ar_model_logits(keye_ar_model, inputs, ar_hook)

        # 获取两个模型的logits
        print("获取KeyeForConditionalGeneration的logits...")
        keye_conditional_logits = get_keye_conditional_generation_logits(keye_conditional_model, inputs, conditional_hook)
        
        # 比较logits
        logits_success = compare_logits(
            keye_conditional_logits.reshape(keye_ar_logits.shape), 
            keye_ar_logits, 
            "KeyeForConditionalGeneration", 
            "KeyeARModel",
            tolerance=1e-4  # 设置容差
        )
        
        # 比较层输出
        layer_success, layer_comparisons = compare_layer_outputs(conditional_hook, ar_hook, tolerance=1e-4)
        
        # 移除hook
        conditional_hook.remove_hooks()
        ar_hook.remove_hooks()
        
        # 输出最终结果
        print("\n" + "="*80)
        print("最终验证结果")
        print("="*80)
        print(f"Logits一致性: {'✅ 通过' if logits_success else '❌ 失败'}")
        print(f"层对齐一致性: {'✅ 通过' if layer_success else '❌ 失败'}")
        
        if logits_success and layer_success:
            print("\n🎉 验证成功！两个模型的前向logits和层输出完全一致。")
        else:
            print("\n❌ 验证失败！请检查上述报告。")
            return 1
            
    except Exception as e:
        print(f"❌ 验证过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())