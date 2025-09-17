import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision.transforms import Resize
from transformers import AutoConfig, AutoModel
# from transformers import Siglip2VisionConfig, Siglip2VisionModel

from . import models
from .utils import ScalingLayer

from huggingface_hub import hf_hub_download

# ckpt_path = hf_hub_download(
#     repo_id="csuhan/TA-Tok",
#     filename="ta_tok.pth",
#     repo_type="model"     
# )
# print("ckpt_path: ", ckpt_path) # '/root/.cache/huggingface/hub/models--csuhan--TA-Tok/snapshots/9a538b78eaab55e5c6b655bbfe177bbd7d581ed3/ta_tok.pth'
ckpt_path = "/mmu_mllm_hdd_2/weimuhao/model/tatok/ta_tok.pth"


class TokenDecoder(nn.Module):
    def __init__(self, hidden_dim, depth=3):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                batch_first=True
            )
            for _ in range(depth)
        ])
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, attn_mask=None):
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=attn_mask)
        return self.out_proj(x)  # [b, n, c]

class TextAlignedTokenizer(nn.Module):
    def __init__(
        self, 
        visual_encoder,
        decoder_config,
        bottleneck,
        bottleneck_token_num=256,
        input_size=384,
        llm_model=None,
        teacher='google/siglip2-so400m-patch14-384',
        input_type='quant', # choose from ['quant', 'rec', 'indices']
        pool_scale=1, # choose from [1, 2, 3]
        decoder_depth=3,
        select_layer_id=-2,
        *args,
        **kwargs
    ):
        super().__init__()
        self.input_size = input_size
        self.bottleneck_token_num = bottleneck_token_num
        self.teacher = teacher
        self.input_type = input_type
        self.pool_scale = pool_scale
        self.decoder_depth = decoder_depth
        self.select_layer_id = select_layer_id

        self.bottleneck_dim = bottleneck['args']['bottleneck_dim'] # self.bottleneck_dim = visual_encoder

        print("self.bottleneck_dim: ", self.bottleneck_dim)

        self.reconstruction_type = 'cosine'

        # TODO: decoder init
        self.encoder_hidden_dim = visual_encoder # 这里的 visual_encoder 是 input_dim 

        # TODO: decoder dim???
        self.decoder = TokenDecoder(hidden_dim=self.encoder_hidden_dim, depth=self.decoder_depth)


        self.encode_task_layer = nn.Sequential(
            nn.Linear(self.encoder_hidden_dim, self.encoder_hidden_dim), # 4096
            nn.Tanh())

        self.decode_task_layer = nn.Sequential(
            nn.Linear(self.encoder_hidden_dim, self.encoder_hidden_dim),
            nn.Tanh(),
            nn.Linear(self.encoder_hidden_dim, self.encoder_hidden_dim))
        

        bottleneck_args = {
            'llm_model': llm_model,
            'token_nums': self.bottleneck_token_num, 
            'input_dim': self.encoder_hidden_dim, 
            'output_dim': self.bottleneck_dim
        }

        self.bottleneck = models.make(bottleneck, args=bottleneck_args) # vector quantization
        self.bottleneck.train()

        # self.scale_layer = ScalingLayer(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])   
        self.image_resize = Resize((self.input_size, self.input_size))

        self.l2_normalized = True
        self.train(True)

    '''
    def set_trainable_modules(self):
        # 先全部冻结
        for p in self.parameters():
            p.requires_grad = False

        # 再解冻需要训练的部分
        for module in [self.bottleneck, self.decoder, self.encode_task_layer, self.decode_task_layer]:
            for p in module.parameters():
                p.requires_grad = True
    '''

    def set_vq_eval_deterministic(self, deterministic=True):
        self.bottleneck.regularizer.set_eval_deterministic(deterministic)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype
    
    '''
    @classmethod
    def from_checkpoint(cls, visual_encoder, decoder_config, load_teacher=True, **kwargs):
        
        
        # ckpt = torch.load(ckpt_path, map_location='cpu')
        # ckpt_kwargs = ckpt["model"]["args"]
        # visual_encoder = 4096 = token embedding dim
        
        ckpt_kwargs = {'bottleneck': {'name': 'bottleneck', 'args': {'bottleneck_dim': visual_encoder, 'norm': 'none', 'regularizer': {'name': 'simvq', 'args': {'codebook_size': 65536, 'commitment_loss_weight': 0.25, 'codebook_loss_weight': 1.0, 'entropy_loss_weight': 0.0, 'entropy_loss_temperature': 0.01, 'l2_normalized': True, 'stochastic': True, 'stochastic_temperature': 0.03, 'top_k': 4, 'top_k_prob': 0.5, 'residual_weight': 0.1}}}}, 'bottleneck_token_num': 729, 'input_size': 384, 'teacher': 'google/siglip2-so400m-patch14-384', 'ckpt_path': 'google/siglip2-so400m-patch14-384', 'pool_scale': 1, 'rand_scale': True}
        model = cls(visual_encoder=visual_encoder, decoder_config=decoder_config, **kwargs, **ckpt_kwargs) # __init__


        # sd = ckpt["model"]["sd"]
        # if not load_teacher:
        #     sd = {k: v for k, v in sd.items() if not k.startswith('teacher')}
        # model.load_state_dict(sd, strict=True)
        return model
    '''

    def encode(self, x, **kwargs):
        # if x.ndim == 5:
        #     x = rearrange(x, 'b c t h w -> (b t) c h w')
        
        # x = self.scale_layer(x)
        # if tuple(x.shape[-2:]) != (self.input_size, self.input_size):
        #     x = self.image_resize(x)
        # vq_feats = self.encoder(x, output_hidden_states=True).hidden_states[self.select_layer_id] 
        # TODO: 
        # x = x.detach()       # 避免梯度回传到 ViT（可选）
        # x.requires_grad_(True)      # 允许后续 decoder 计算梯度
        

        
        vq_feats = x # (b, n,c)

        # pooling not work
        pool_scale = self.pool_scale
        pool_scale = kwargs.get("pool_scale", pool_scale)
        # if pool_scale != 1:
        #     vq_feats = self.avg_pool(vq_feats, pool_scale)
        print("vq_feats:", vq_feats.shape) # torch.Size([1, 1500, 4096])
        vq_feats = self.encode_task_layer(vq_feats.to(x.device)) # (b, n, c)
        print("vq_feats.requires_grad:", vq_feats.requires_grad)
        bottleneck_out = self.bottleneck(vq_feats)
        z = bottleneck_out.pop('output') # quantized

        return {'encoded': z, 'pool_scale': pool_scale, 'vq_feats': vq_feats, **bottleneck_out}

    def avg_pool(self, z, pool_scale=1):
        if z.ndim == 3:
            b, n, c = z.shape
            p = int(n ** 0.5)
            z = rearrange(z, 'b (p1 p2) c -> b c p1 p2', p1=p, p2=p)
        else:
            b, c, p, _ = z.shape
        p_s = int(p // pool_scale)
        z = F.avg_pool2d(
            z,
            kernel_size=(pool_scale, pool_scale),
            stride=(pool_scale, pool_scale)
        ).contiguous()
        z = rearrange(z, 'b c p1 p2 -> b (p1 p2) c')
        return z

    def decode(self, z, attn_mask):
        # if z.ndim == 4:
        #     z = rearrange(z, 'b c p1 p2 -> b (p1 p2) c')
        # attention_mask = torch.ones(z.shape[:2], dtype=torch.int, device=z.device)
        # p = int(z.shape[1]**0.5)
        # spatial_shape = torch.tensor([[p, p]]*z.shape[0], device=self.device)
        # z = self.decoder(z, attention_mask, spatial_shape, output_hidden_states=True).last_hidden_state
        # z = self.decode_task_layer(z)
        # TODO: decode model
        z = self.decoder(z, attn_mask)
        z = self.decode_task_layer(z)
        print("z: after decoder: ", z.shape)

        return z

    def decode_from_bottleneck(self, bottleneck_rep):
        z = self.bottleneck.decode(bottleneck_rep) # (b, n, c)
        p = int(z.shape[1]**0.5)
        z = rearrange(z, 'b (p1 p2) c -> b c p1 p2', p1=p, p2=p)
        return self.decode(z)

    def forward(self, data, teacher_data, eps=1e-8, **kwargs):
        # data: video in shape (b, c, t, h, w)
        # data: image in shape (b, n, c)
        encode_output = self.encode(data, **kwargs)
        vq_feats = encode_output['encoded'] # quantized
        print("vq_feats after self.encode(quantized):", vq_feats.shape) # torch.Size([1, 414, 4096])
        # p = int(vq_feats.shape[1] ** 0.5)
        # vq_feats = rearrange(vq_feats, 'b (h w) c -> b c h w', h=p, w=p)
        # TODO: decode model
        print("self.input_type: ", self.input_type)
        # pred_feats = self.decode(vq_feats)
        attn_mask = None
        pred_feats = self.decode(vq_feats, attn_mask=attn_mask)
        # TODO: no decoder
        # pred_feats = vq_feats

        # self.input_type: quant
        
        if self.input_type == 'quant':
            z = encode_output["regularized_z"] # [b, n, c] quantized
        elif self.input_type == 'indices':
            z = encode_output["bottleneck_rep"] # [b, n] indices
        elif self.input_type == 'rec':
            z = pred_feats # [b, n, c] rec
        encode_output['encoded'] = z


        if self.reconstruction_type == "mse":
            if attn_mask is not None:  # mask 掉 padding位置
                reconstruction_loss = ((teacher_data - pred_feats) ** 2) * attn_mask[..., None]
                reconstruction_loss = reconstruction_loss.sum() / attn_mask.sum()
            else:
                if self.l2_normalized:
                    print("teacher_data l2_normalized!")
                    teacher_data = F.normalize(teacher_data, p=2, dim=-1)
                reconstruction_loss = F.mse_loss(pred_feats, teacher_data)
        else: # cosine similarity
            pred_norm = pred_feats / (pred_feats.norm(dim=-1, keepdim=True) + eps)
            teacher_norm = teacher_data / (teacher_data.norm(dim=-1, keepdim=True) + eps)

            cos_sim = (pred_norm * teacher_norm).sum(dim=-1)  # [B, T]
            cos_loss = 1 - cos_sim  # [B, T]
            print("cos_loss:", cos_loss.shape)

            if attn_mask is not None:
                cos_loss = cos_loss * attn_mask  # mask 掉 padding
                reconstruction_loss = cos_loss.sum() / attn_mask.sum()
            else:
                reconstruction_loss = cos_loss.mean()


        print("reconstruction_loss: ", reconstruction_loss)
        encode_output['reconstruction_loss'] = reconstruction_loss
        print("reconstruction_loss.requires_grad:", reconstruction_loss.requires_grad)
        return encode_output
