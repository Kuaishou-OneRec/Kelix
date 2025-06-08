import random
import torch
from PIL import Image
import numpy as np
from omegaconf import OmegaConf
from recipes.ViT.training.models import KimiViT, KimiViTSigLIP


config = OmegaConf.load("/llm_reco/zangdunju/vllm/vit/recovlm/recipes/ViT/configs/v1.yaml")
config.model.packing = config.dataset.packing


class Context(dict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)

    def __getitem__(self, key):
        return super().__getitem__(key)

    def __setattr__(self, key, value):
        super().__setitem__(key, value)

    def __getattr__(self, key):
        return super().__getitem__(key)


class DistributedContext(Context):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self):
        self["world_size"] = 1
        self["rank"] = 0
        self["is_dist"] = True
        return self

seed = 1234
random.seed(seed)
    
np.random.seed(seed)
    
torch.manual_seed(seed)
    
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
        
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

arr = torch.randint(0, 255, (224, 224, 3), dtype=torch.uint8).numpy()
image = Image.fromarray(arr)
image = image.convert("RGB")

ctx = DistributedContext().setup()
model = KimiViT(config.model, ctx).to(torch.bfloat16).cuda()
processor = model.processor
pixel_values = processor(images=[image], do_resize=False, return_tensors="pt")["pixel_values"]
pixel_values = pixel_values.squeeze(0)
from einops import rearrange
pixel_values = rearrange(pixel_values, "c (h p1) (w p2) -> (h w) c p1 p2", p1=14, p2=14).cuda()
print(pixel_values.shape)
print(model.__class__.__name__)


package = Context(
    images=[image],
    texts=["empty image."],
    source=["empty"],
    pixel_values=pixel_values[None, :, :, :, :],
    image_position_ids=torch.arange(256).cuda()
)

model(package=package, images=None, texts=None)


