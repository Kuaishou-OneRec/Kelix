from transformers import AutoProcessor, AutoModel, AutoConfig
import torch
import os
from safetensors.torch import save_file
import subprocess
import torch.nn.functional as F
import copy

from MoonVision.modeling_kimi_vl import MoonVitPretrainedModel
from MoonVision.image_processing_kimi_vl import KimiVLImageProcessor
from MoonVision.configuration_kimi_vl import MoonViTConfig, KimiVLConfig

def merge_qkv_parameters(state_dict, num_layers=27):
    """
    将分开的Q、K、V投影层参数合并为wqkv格式
    
    参数:
        state_dict: 原始模型的状态字典
        num_layers: 模型的层数(默认为27)
        
    返回:
        合并后的新状态字典
    """
    new_state_dict = {}
    
    # 首先复制所有不需要修改的参数
    for key in state_dict:
        if not any(f'encoder.layers.{i}.self_attn' in key for i in range(num_layers)):
            new_state_dict[key] = state_dict[key]
    
    # 处理每一层的QKV参数
    for layer in range(num_layers):
        # 获取Q、K、V的权重和偏置
        q_weight = state_dict[f'encoder.layers.{layer}.self_attn.q_proj.weight']
        k_weight = state_dict[f'encoder.layers.{layer}.self_attn.k_proj.weight'] 
        v_weight = state_dict[f'encoder.layers.{layer}.self_attn.v_proj.weight']
        
        q_bias = state_dict[f'encoder.layers.{layer}.self_attn.q_proj.bias']
        k_bias = state_dict[f'encoder.layers.{layer}.self_attn.k_proj.bias']
        v_bias = state_dict[f'encoder.layers.{layer}.self_attn.v_proj.bias']
        
        # 合并权重 (按照Q、K、V的顺序拼接)
        wqkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
        new_state_dict[f'encoder.layers.{layer}.self_attn.wqkv.weight'] = wqkv_weight
        
        # 合并偏置
        wqkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
        new_state_dict[f'encoder.layers.{layer}.self_attn.wqkv.bias'] = wqkv_bias
        
    return new_state_dict

if __name__ == '__main__':
    siglip_path = '/llm_reco/liuyang76/Models/siglip2-so400m-patch14-384'
    MoonVit_path = '/llm_reco/liuyang76/Models/Kimi-VL-A3B-Instruct/'
    siglip2 = AutoModel.from_pretrained(siglip_path ,trust_remote_code=True).vision_model.state_dict()
    # MoonVit = AutoModel.from_pretrained(MoonVit_path ,trust_remote_code=True)
    MoonViT_config = MoonViTConfig()
    MoonVit = MoonVitPretrainedModel(MoonViT_config).state_dict()

    siglip2 = merge_qkv_parameters(siglip2)
    vit_save_dir = '/llm_reco/liuyang76/Models/MoonVitParam'

    for name, param in siglip2.items():
        # for encoder
        if name.startswith('encoder'):
            MoonVitp_name = name.replace('.layers', '.blocks')\
                                .replace('.layer_norm2', '.norm1')\
                                .replace('.layer_norm1', '.norm0')\
                                .replace('.fc1.', '.fc0.')\
                                .replace('.fc2.', '.fc1.')\
                                .replace('.self_attn.out_proj.', '.wo.')\
                                .replace('.self_attn.wqkv.', '.wqkv.')
            MoonVit[MoonVitp_name] = param
            # continue
        elif name == 'embeddings.position_embedding.weight':
            MoonVitp_name = name.replace('embeddings.position_embedding.', 'patch_embed.pos_emb.')
            grid_size = [64, 64]
            interpolation='nearest'
            antialias=False
            align_corners=None
            pos_emb_img = param
            pos_emb_img = pos_emb_img.reshape(1, 27, 27, -1).permute(0, 3, 1, 2)
            pos_emb_img = F.interpolate(
                pos_emb_img,
                size=grid_size,
                mode=interpolation,
                antialias=antialias,
                align_corners=align_corners,
            )
            pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], -1)[0]
            MoonVit[MoonVitp_name] = pos_emb_img.reshape(64, 64, -1)
            # continue

        elif name.startswith('embeddings'):
            MoonVitp_name = name.replace('embeddings.patch_embedding.', 'patch_embed.proj.')
            MoonVit[MoonVitp_name] = param
            # continue
        else:
            continue
    save_file(MoonVit, os.path.join(vit_save_dir, 'model.safetensors'), metadata={'format': 'pt'})
    torch.save(MoonVit, os.path.join(vit_save_dir, 'MoonVit.pt'))