import os
import torch
from torch import nn
from transformers.models.siglip.image_processing_siglip import SiglipImageProcessor

from .utils import rank0_print
from ..tok.ta_tok import TextAlignedTokenizer
from ..tok.utils import ScalingLayer


class TATokVisionTower(nn.Module):
    def __init__(self, vision_tower, vision_tower_cfg, visual_encoder, decoder_config, llm_model, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.config = None

        self.image_processor = SiglipImageProcessor()

        self.vision_tower_name = vision_tower
        
        if not delay_load:
            rank0_print(f"Loading vision tower: {vision_tower}")
            self.load_model(visual_encoder=visual_encoder, decoder_config=decoder_config, llm_model=llm_model)
        elif getattr(vision_tower_cfg, "unfreeze_mm_vision_tower", False):
            # TODO: better detector is needed.
            rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `unfreeze_mm_vision_tower`: True.")
            self.load_model(visual_encoder=visual_encoder, decoder_config=decoder_config, llm_model=llm_model)
        elif hasattr(vision_tower_cfg, "mm_tunable_parts") and "mm_vision_tower" in vision_tower_cfg.mm_tunable_parts:
            rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `mm_tunable_parts` contains `mm_vision_tower`.")
            self.load_model(visual_encoder=visual_encoder, decoder_config=decoder_config, llm_model=llm_model)
        else:
            self.cfg_only = self.config
        
        self.train(True)

    def load_model(self, visual_encoder, decoder_config, llm_model, device_map=None):
        if self.is_loaded:
            rank0_print("{} is already loaded, `load_model` called again, skipping.".format(self.vision_tower_name))
            return

        ckpt_kwargs = {
            'bottleneck': {
                'name': 'bottleneck', 
                'args': {
                    # 'bottleneck_dim': visual_encoder, 
                    'bottleneck_dim': 1536,
                    'norm': 'none', 
                    'regularizer': {
                        'name': 'simvq', 
                        'args': {
                            'codebook_size': 65536, 
                            'commitment_loss_weight': 1.0, 
                            'codebook_loss_weight': 1.0, 
                            'entropy_loss_weight': 1, 
                            'entropy_loss_temperature': 1, 
                            'l2_normalized': False, 
                            'stochastic': False, 
                            'stochastic_temperature': 0.3, 
                            'top_k': 10, 
                            'top_k_prob': 0.5, 
                            'residual_weight': 0.2}}}}, 
            'bottleneck_token_num': 729, 
            'input_size': 384, 
            'teacher': 'google/siglip2-so400m-patch14-384', 
            'ckpt_path': 'google/siglip2-so400m-patch14-384', 
            'pool_scale': 1, 
            'rand_scale': True}

            
        self.vision_tower = TextAlignedTokenizer(visual_encoder=visual_encoder, decoder_config=decoder_config, load_teacher=False, llm_model=llm_model, **ckpt_kwargs).to(device_map) # __init__
        
        # TODO:
        # self.vision_tower.bottleneck.regularizer.set_eval_deterministic(deterministic=True)

        # self.vision_tower.input_type = 'rec'
        self.vision_tower.input_type = 'quant'

        # self.vision_tower.scale_layer = ScalingLayer(mean=[0., 0., 0.], std=[1., 1., 1.])

        # TODO: 
        # self.vision_tower.set_trainable_modules()
        # self.vision_tower.requires_grad_(False)
        # self.vision_tower.eval()
        # self.vision_tower.train()

        self.pool_scales = [1, 1, 2, 3]

        input_size = self.vision_tower.input_size
        self.image_processor.size = (input_size, input_size)
        self.image_processor.crop_size = {'height': input_size, 'width': input_size}

        self.image_tokens = self.vision_tower.bottleneck_token_num
        self.bottleneck_dim = self.vision_tower.bottleneck_dim
        self.num_patches = self.image_tokens
        self.num_patches_per_side = int(self.num_patches ** 0.5)
        self.hidden_size = self.vision_tower.encoder_hidden_dim
        self.image_size = input_size

        self.is_loaded = True

    def get_embedding(self):
        return self.vision_tower.bottleneck.regularizer.get_emb()
    
    def forward(self, images, teacher_image_embeds, pool_scale=1):
        # load from ENV
        # pool_scale from ENV has the highest priority
        pool_scale = int(os.environ.get('POOL_SCALE', pool_scale))

        if pool_scale is None: pool_scale = 1
        if type(images) is list:
            image_features, tokens = [], []
            image_forward_outs = []
            for image, teacher_image in zip(images, teacher_image_embeds):
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), teacher_image.to(device=self.device, dtype=self.dtype).unsqueeze(0), pool_scale=pool_scale)
                image_feature, token = image_forward_out['vq_feats'].to(image.dtype), image_forward_out['bottleneck_rep']
                image_features.append(image_feature)
                tokens.append(token)
                image_forward_outs.append(image_forward_out)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), pool_scale=pool_scale)
            image_features, tokens = image_forward_outs['vq_feats'].to(images.dtype), image_forward_outs['bottleneck_rep']
        return {"image_features": image_features, "tokens": tokens, 'pool_scale': pool_scale, "image_forward_outs":image_forward_outs} # TODO:

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        for p in self.vision_tower.parameters():
            return p.dtype

    @property
    def device(self):
        for p in self.vision_tower.parameters():
            return p.device