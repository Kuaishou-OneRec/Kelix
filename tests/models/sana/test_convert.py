from safetensors.torch import load_file, save_file
import glob
import tqdm
import torch
import re
import os

def load_full_safe_tensors(paths):
    """加载完整的Safetensors（保留tensor数值）"""
    ckpt = {}
    for path in tqdm.tqdm(paths, desc="Loading old CKPT"):
        ckpt.update(load_file(path))
    return ckpt

def convert_old_to_new_ckpt(old_ckpt, device='cpu'):
    """
    精准映射旧CKPT到新CKPT state dict
    仅做参数名转换，不随机初始化任何缺失参数，不新增额外参数
    """
    new_ckpt = {}

    # -------------------------- 精准参数映射规则（逐参数对齐） --------------------------
    replace_rules = [
        # 1. Patch Embedding: sana.patch_embed.proj → x_embedder.proj
        ('^sana.patch_embed.proj.bias$', 'x_embedder.proj.bias'),
        ('^sana.patch_embed.proj.weight$', 'x_embedder.proj.weight'),

        # 2. Time Embedding: sana.time_embed → t_embedder + t_block
        ('^sana.time_embed.emb.timestep_embedder.linear_1.bias$', 't_embedder.mlp.0.bias'),
        ('^sana.time_embed.emb.timestep_embedder.linear_1.weight$', 't_embedder.mlp.0.weight'),
        ('^sana.time_embed.emb.timestep_embedder.linear_2.bias$', 't_embedder.mlp.2.bias'),
        ('^sana.time_embed.emb.timestep_embedder.linear_2.weight$', 't_embedder.mlp.2.weight'),
        ('^sana.time_embed.linear.bias$', 't_block.1.bias'),
        ('^sana.time_embed.linear.weight$', 't_block.1.weight'),

        # 3. 全局Scale Shift Table: sana.scale_shift_table → final_layer.scale_shift_table
        ('^sana.scale_shift_table$', 'final_layer.scale_shift_table'),

        # 4. 输出投影: sana.proj_out → final_layer.linear
        ('^sana.proj_out.bias$', 'final_layer.linear.bias'),
        ('^sana.proj_out.weight$', 'final_layer.linear.weight'),

        # 5. 文本投影+归一化: sana.caption_* → y_embedder.y_proj + attention_y_norm
        ('^sana.caption_norm.weight$', 'attention_y_norm.weight'),
        ('^sana.caption_projection.linear_1.bias$', 'y_embedder.y_proj.fc1.bias'),
        ('^sana.caption_projection.linear_1.weight$', 'y_embedder.y_proj.fc1.weight'),
        ('^sana.caption_projection.linear_2.bias$', 'y_embedder.y_proj.fc2.bias'),
        ('^sana.caption_projection.linear_2.weight$', 'y_embedder.y_proj.fc2.weight'),

        # 6. Transformer Blocks: sana.transformer_blocks[i] → blocks[i]
        # 6.1 自注意力层（attn1 → attn）
        ('^sana.transformer_blocks.([0-9]+).attn1.norm_k.weight$', 'blocks.\\1.attn.k_norm.weight'),
        ('^sana.transformer_blocks.([0-9]+).attn1.norm_q.weight$', 'blocks.\\1.attn.q_norm.weight'),
        ('^sana.transformer_blocks.([0-9]+).attn1.to_k.weight$', 'blocks.\\1.attn.to_k.weight'),
        ('^sana.transformer_blocks.([0-9]+).attn1.to_out.0.bias$', 'blocks.\\1.attn.proj.bias'),
        ('^sana.transformer_blocks.([0-9]+).attn1.to_out.0.weight$', 'blocks.\\1.attn.proj.weight'),
        ('^sana.transformer_blocks.([0-9]+).attn1.to_q.weight$', 'blocks.\\1.attn.to_q.weight'),
        ('^sana.transformer_blocks.([0-9]+).attn1.to_v.weight$', 'blocks.\\1.attn.to_v.weight'),

        # 6.2 交叉注意力层（attn2 → cross_attn）
        ('^sana.transformer_blocks.([0-9]+).attn2.norm_k.weight$', 'blocks.\\1.cross_attn.k_norm.weight'),
        ('^sana.transformer_blocks.([0-9]+).attn2.norm_q.weight$', 'blocks.\\1.cross_attn.q_norm.weight'),
        ('^sana.transformer_blocks.([0-9]+).attn2.to_k.bias$', 'blocks.\\1.cross_attn.to_k.bias'),
        ('^sana.transformer_blocks.([0-9]+).attn2.to_k.weight$', 'blocks.\\1.cross_attn.to_k.weight'),
        ('^sana.transformer_blocks.([0-9]+).attn2.to_out.0.bias$', 'blocks.\\1.cross_attn.proj.bias'),
        ('^sana.transformer_blocks.([0-9]+).attn2.to_out.0.weight$', 'blocks.\\1.cross_attn.proj.weight'),
        ('^sana.transformer_blocks.([0-9]+).attn2.to_q.bias$', 'blocks.\\1.cross_attn.q_linear.bias'),
        ('^sana.transformer_blocks.([0-9]+).attn2.to_q.weight$', 'blocks.\\1.cross_attn.q_linear.weight'),
        ('^sana.transformer_blocks.([0-9]+).attn2.to_v.bias$', 'blocks.\\1.cross_attn.to_v.bias'),
        ('^sana.transformer_blocks.([0-9]+).attn2.to_v.weight$', 'blocks.\\1.cross_attn.to_v.weight'),

        # 6.3 前馈网络（ff → mlp）
        ('^sana.transformer_blocks.([0-9]+).ff.conv_depth.bias$', 'blocks.\\1.mlp.depth_conv.conv.bias'),
        ('^sana.transformer_blocks.([0-9]+).ff.conv_depth.weight$', 'blocks.\\1.mlp.depth_conv.conv.weight'),
        ('^sana.transformer_blocks.([0-9]+).ff.conv_inverted.bias$', 'blocks.\\1.mlp.inverted_conv.conv.bias'),
        ('^sana.transformer_blocks.([0-9]+).ff.conv_inverted.weight$', 'blocks.\\1.mlp.inverted_conv.conv.weight'),
        ('^sana.transformer_blocks.([0-9]+).ff.conv_point.weight$', 'blocks.\\1.mlp.point_conv.conv.weight'),

        # 6.4 Block内Scale Shift Table
        ('^sana.transformer_blocks.([0-9]+).scale_shift_table$', 'blocks.\\1.scale_shift_table'),
    ]

    # -------------------------- 应用映射规则 --------------------------
    for old_key, old_tensor in old_ckpt.items():
        # 跳过不需要的参数（与原加载逻辑一致）
        if 'visual' in old_key or 'vae' in old_key or 'model.layers.' in old_key:
            continue

        new_key = None
        # 精准匹配每个参数
        for pattern, replacement in replace_rules:
            if re.fullmatch(pattern, old_key):
                new_key = re.sub(pattern, replacement, old_key)
                break

        # 保存映射后的参数
        if new_key:
            new_ckpt[new_key] = old_tensor.to(device)
        # diffusion_connector参数名完全一致，直接保留
        elif old_key.startswith('diffusion_connector.'):
            new_ckpt[old_key] = old_tensor.to(device)
        # 未匹配参数提示（便于排查）
        else:
            print(f"⚠️  未匹配的旧参数（已忽略）: {old_key}")

    return new_ckpt

def save_new_ckpt(new_ckpt, save_dir, num_shards=4):
    """保存为4分片Safetensors（与新CKPT格式一致）"""
    os.makedirs(save_dir, exist_ok=True)
    keys = sorted(list(new_ckpt.keys()))  # 排序保证分片稳定
    shard_size = (len(keys) + num_shards - 1) // num_shards

    for shard_idx in range(num_shards):
        start = shard_idx * shard_size
        end = min((shard_idx + 1) * shard_size, len(keys))
        shard_ckpt = {k: new_ckpt[k] for k in keys[start:end]}
        
        shard_filename = f"model-{shard_idx:05d}-of-0000{num_shards}.safetensors"
        save_path = os.path.join(save_dir, shard_filename)
        save_file(shard_ckpt, save_path)
        print(f"✅ 保存分片: {save_path}")

def main():
    # 配置路径
    old_ckpt_paths = glob.glob(
        '/mmu_mllm_hdd_2/zangdunju/output2/RecoVLM/unify/blip3o_tfr_sft/3.0.2/step200/global_step200/converted/*.safetensors'
    )
    new_ckpt_save_dir = "/path/to/save/new_ckpt"  # 请修改为实际保存路径

    # 执行转换
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔄 开始CKPT转换（设备: {device}）")

    # 1. 加载旧CKPT
    old_ckpt = load_full_safe_tensors(old_ckpt_paths)
    print(f"📥 加载完成：旧CKPT共{len(old_ckpt)}个参数")

    # 2. 格式转换（仅参数名映射，无随机初始化）
    new_ckpt = convert_old_to_new_ckpt(old_ckpt, device=device)
    print(f"📊 转换完成：新CKPT共{len(new_ckpt)}个参数")

    # 3. 保存新CKPT
    save_new_ckpt(new_ckpt, new_ckpt_save_dir)
    print(f"\n📤 新CKPT保存至：{new_ckpt_save_dir}")

    # 关键配置提示（必须执行）
    print("\n⚠️  转换后配置文件修改要求：")
    print("1. attention_y_norm.scale = 1.0（覆盖默认0.01，对齐旧逻辑）")
    print("2. 关闭cross_attn_x_norm（设置为false）")
    print("3. 确认y_embedder为2层MLP结构")

if __name__ == "__main__":
    main()