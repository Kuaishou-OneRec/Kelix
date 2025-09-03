from .ta_tok_encoder import TATokVisionTower
import torch

# def build_vision_tower(vision_tower_cfg, **kwargs):
#     vision_tower = getattr(vision_tower_cfg, "mm_vision_tower", getattr(vision_tower_cfg, "vision_tower", None))
#     # return TATok by default, you can add more tokenizers here   
#     return TATokVisionTower(vision_tower, vision_tower_cfg=vision_tower_cfg, **kwargs)


def build_vision_tower(vision_tower_name, vision_tower_cfg, visual_encoder, decoder_config, delay_load):
    return TATokVisionTower(vision_tower=vision_tower_name, vision_tower_cfg=vision_tower_cfg, visual_encoder=visual_encoder, decoder_config=decoder_config, delay_load=delay_load)
    