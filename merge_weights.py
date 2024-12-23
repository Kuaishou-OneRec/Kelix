import argparse
import re
import os

import torch
import transformers
# Qwen2VLForConditionalGeneration

def get_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", type=str, default=None,
                        help="The directory of the pretrained LLM.")

    parser.add_argument("--vision_encoder_dir", type=str, default=None,
                        help="The directory of the pretrained ViT.")

    parser.add_argument("--new_model_dir", type=str, default=None,
                        help="The directory of the pretrained ViT.")

    return parser

# embeddings.class_embedding torch.Size([1280])
# embeddings.patch_embedding.weight torch.Size([1280, 3, 14, 14])
# embeddings.position_embedding.weight torch.Size([730, 1280])
# pre_layrnorm.weight torch.Size([1280])
# pre_layrnorm.bias torch.Size([1280])
# encoder.layers.0.self_attn.k_proj.weight torch.Size([1280, 1280])
# encoder.layers.0.self_attn.k_proj.bias torch.Size([1280])
# encoder.layers.0.self_attn.v_proj.weight torch.Size([1280, 1280])
# encoder.layers.0.self_attn.v_proj.bias torch.Size([1280])
# encoder.layers.0.self_attn.q_proj.weight torch.Size([1280, 1280])
# encoder.layers.0.self_attn.q_proj.bias torch.Size([1280])
# encoder.layers.0.self_attn.out_proj.weight torch.Size([1280, 1280])
# encoder.layers.0.self_attn.out_proj.bias torch.Size([1280])
# encoder.layers.0.layer_norm1.weight torch.Size([1280])
# encoder.layers.0.layer_norm1.bias torch.Size([1280])
# encoder.layers.0.mlp.fc1.weight torch.Size([5120, 1280])
# encoder.layers.0.mlp.fc1.bias torch.Size([5120])
# encoder.layers.0.mlp.fc2.weight torch.Size([1280, 5120])
# encoder.layers.0.mlp.fc2.bias torch.Size([1280])
# encoder.layers.0.layer_norm2.weight torch.Size([1280])
# encoder.layers.0.layer_norm2.bias torch.Size([1280])
# post_layernorm.weight torch.Size([1280])
# post_layernorm.bias torch.Size([1280])


# patch_embed.proj.weight torch.Size([1280, 3, 2, 14, 14])

# blocks.0.norm1.weight torch.Size([1280])
# blocks.0.norm1.bias torch.Size([1280])
# blocks.0.norm2.weight torch.Size([1280])
# blocks.0.norm2.bias torch.Size([1280])
# blocks.0.attn.qkv.weight torch.Size([3840, 1280])
# blocks.0.attn.qkv.bias torch.Size([3840])
# blocks.0.attn.proj.weight torch.Size([1280, 1280])
# blocks.0.attn.proj.bias torch.Size([1280])
# blocks.0.mlp.fc1.weight torch.Size([5120, 1280])
# blocks.0.mlp.fc1.bias torch.Size([5120])
# blocks.0.mlp.fc2.weight torch.Size([1280, 5120])
# blocks.0.mlp.fc2.bias torch.Size([1280])

# merger.ln_q.weight torch.Size([1280])
# merger.ln_q.bias torch.Size([1280])
# merger.mlp.0.weight torch.Size([5120, 5120])
# merger.mlp.0.bias torch.Size([5120])
# merger.mlp.2.weight torch.Size([3584, 5120])
# merger.mlp.2.bias torch.Size([3584])

def main():
    arg_parser = get_argument_parser()
    args = arg_parser.parse_args()

    # llm weights
    model = transformers.Qwen2VLForConditionalGeneration.from_pretrained(args.model_dir)
    sd = model.state_dict()
    if args.vision_encoder_dir:
        vision_encoder = transformers.CLIPModel.from_pretrained(args.vision_encoder_dir)

        vision_sd = vision_encoder.vision_model.state_dict()
        mapped = [
            "post_layernorm.weight",
            "post_layernorm.bias",
            "embeddings.patch_embedding.weight"
        ]
        assert "visual.merger.ln_q.weight" in sd
        sd["visual.merger.ln_q.weight"] = vision_sd["post_layernorm.weight"]
        assert "visual.merger.ln_q.bias" in sd
        sd["visual.merger.ln_q.bias"] = vision_sd["post_layernorm.bias"]
        assert "visual.patch_embed.proj.weight" in sd
        # conv2d -> 3d
        sd["visual.patch_embed.proj.weight"] = vision_sd["embeddings.patch_embedding.weight"][:,:,None,:,:].repeat(1, 1, 2, 1, 1)
        for layer in range(vision_encoder.vision_model.config.num_hidden_layers):
            wq = vision_sd[f"encoder.layers.{layer}.self_attn.q_proj.weight"]
            wk = vision_sd[f"encoder.layers.{layer}.self_attn.k_proj.weight"]
            wv = vision_sd[f"encoder.layers.{layer}.self_attn.v_proj.weight"]

            bq = vision_sd[f"encoder.layers.{layer}.self_attn.q_proj.bias"]
            bk = vision_sd[f"encoder.layers.{layer}.self_attn.k_proj.bias"]
            bv = vision_sd[f"encoder.layers.{layer}.self_attn.v_proj.bias"]

            qkv_weight = torch.cat([wq, wk, wv], dim=0)
            qkv_bias = torch.cat([bq, bk, bv], dim=0)

            assert f"visual.blocks.{layer}.attn.qkv.weight" in sd
            sd[f"visual.blocks.{layer}.attn.qkv.weight"] = qkv_weight
            assert f"visual.blocks.{layer}.attn.qkv.bias" in sd
            sd[f"visual.blocks.{layer}.attn.qkv.bias"] = qkv_bias

            assert f"visual.blocks.{layer}.attn.proj.weight" in sd
            sd[f"visual.blocks.{layer}.attn.proj.weight"] = vision_sd[f"encoder.layers.{layer}.self_attn.out_proj.weight"]
            assert f"visual.blocks.{layer}.attn.proj.bias" in sd
            sd[f"visual.blocks.{layer}.attn.proj.bias"] = vision_sd[f"encoder.layers.{layer}.self_attn.out_proj.bias"]
            
            assert f"visual.blocks.{layer}.mlp.fc1.weight" in sd
            sd[f"visual.blocks.{layer}.mlp.fc1.weight"] = vision_sd[f"encoder.layers.{layer}.mlp.fc1.weight"]
            assert f"visual.blocks.{layer}.mlp.fc1.bias" in sd
            sd[f"visual.blocks.{layer}.mlp.fc1.bias"] = vision_sd[f"encoder.layers.{layer}.mlp.fc1.bias"]
            assert f"visual.blocks.{layer}.mlp.fc2.weight" in sd
            sd[f"visual.blocks.{layer}.mlp.fc2.weight"] = vision_sd[f"encoder.layers.{layer}.mlp.fc2.weight"]
            assert f"visual.blocks.{layer}.mlp.fc2.bias" in sd
            sd[f"visual.blocks.{layer}.mlp.fc2.bias"] = vision_sd[f"encoder.layers.{layer}.mlp.fc2.bias"]
            
            assert f"visual.blocks.{layer}.norm1.weight" in sd
            sd[f"visual.blocks.{layer}.norm1.weight"] = vision_sd[f"encoder.layers.{layer}.layer_norm1.weight"]
            assert f"visual.blocks.{layer}.norm1.bias" in sd
            sd[f"visual.blocks.{layer}.norm1.bias"] = vision_sd[f"encoder.layers.{layer}.layer_norm1.bias"]
            assert f"visual.blocks.{layer}.norm2.weight" in sd
            sd[f"visual.blocks.{layer}.norm2.weight"] = vision_sd[f"encoder.layers.{layer}.layer_norm2.weight"]
            assert f"visual.blocks.{layer}.norm2.bias" in sd
            sd[f"visual.blocks.{layer}.norm2.bias"] = vision_sd[f"encoder.layers.{layer}.layer_norm2.bias"]

            mapped.extend([
                f"encoder.layers.{layer}.self_attn.q_proj.weight",
                f"encoder.layers.{layer}.self_attn.k_proj.weight",
                f"encoder.layers.{layer}.self_attn.v_proj.weight",
                f"encoder.layers.{layer}.self_attn.q_proj.bias",
                f"encoder.layers.{layer}.self_attn.k_proj.bias",
                f"encoder.layers.{layer}.self_attn.v_proj.bias",
                f"encoder.layers.{layer}.self_attn.out_proj.weight",
                f"encoder.layers.{layer}.self_attn.out_proj.bias",
                f"encoder.layers.{layer}.mlp.fc1.weight",
                f"encoder.layers.{layer}.mlp.fc1.bias",
                f"encoder.layers.{layer}.mlp.fc2.weight",
                f"encoder.layers.{layer}.mlp.fc2.bias",
                f"encoder.layers.{layer}.layer_norm1.weight",
                f"encoder.layers.{layer}.layer_norm1.bias",
                f"encoder.layers.{layer}.layer_norm2.weight",
                f"encoder.layers.{layer}.layer_norm2.bias"
            ])

        for name in vision_encoder.vision_model.state_dict().keys():
            if not name in mapped:
                print(f"Parameter {name} in VisionEncoder is not mapped.")
    
    model.load_state_dict(sd)
    if not os.path.exists(args.new_model_dir):
        os.makedirs(args.new_model_dir)
    model.save_pretrained(args.new_model_dir)

if __name__ == "__main__":
    main()