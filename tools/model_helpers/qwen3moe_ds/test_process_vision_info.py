
from recovlm.utils.ds_utils import format_dict_or_list
from PIL import Image, ImageDraw
import torch
import torch

from recovlm.training.common import set_default_dtype, get_global_grad_norm, clip_grad_by_value
from recovlm.training.checkpoint import load_hf_checkpoint

from recovlm.utils.qwen_vl_utils import process_vision_info
from recovlm.models.keye.keye_vl_utils import process_vision_info as process_vision_info_keye
# /llm_reco/lingzhixin/recovlm_qw0510/recovlm/recovlm/models/keye_vitrope/keye_vl_utils.py
from PIL import Image
import torch
from recovlm.training.checkpoint import load_hf_checkpoint
from recovlm.training.common import set_default_dtype
from recovlm.models.keye.modeling_keye import KeyeForConditionalGeneration
from recovlm.models.keye.processing_keye import KeyeProcessor



from transformers import AutoProcessor, AutoTokenizer

def set_seed(seed: int):
    import random
    import numpy as np

    """设置所有可能的随机数种子，保证实验可重复性"""
    # 设置 Python 内置的随机数种子
    random.seed(seed)
    # 设置 NumPy 的随机数种子
    np.random.seed(seed)
    # 设置 PyTorch 的 CPU 随机数种子
    torch.manual_seed(seed)
    # 设置 PyTorch 的 CUDA 随机数种子（用于 GPU 计算）
    torch.cuda.manual_seed(seed)
    # 如果使用了多个 GPU，还需要设置这个
    torch.cuda.manual_seed_all(seed)
    # 禁用 CuDNN 的非确定性算法（确保结果可复现）
    torch.backends.cudnn.deterministic = True
    # 禁用 CuDNN 的自动调优功能（确保每次运行使用相同的算法）
    torch.backends.cudnn.benchmark = False

set_seed(0)


torch.cuda.set_device(0)
local_rank = 0


def generate_circle_image(size=(200, 200), fill_color=(0, 0, 0), outline_color=(255, 255, 255), outline_width=5):
    """
    生成一个包含一个圆的 PIL Image 对象。

    :param size: 图像的大小，默认为 (200, 200)
    :param fill_color: 圆的填充颜色，默认为黑色 (0, 0, 0)
    :param outline_color: 圆的轮廓颜色，默认为白色 (255, 255, 255)
    :param outline_width: 圆的轮廓宽度，默认为 5
    :return: 生成的 PIL Image 对象
    """
    # 创建一个新的图像对象
    image = Image.new('RGB', size, color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    # 计算圆的坐标（图像中心为圆心）
    x_center, y_center = size[0] // 2, size[1] // 2
    radius = min(size[0], size[1]) // 2
    # 绘制圆
    draw.ellipse([x_center - radius, y_center - radius, x_center + radius, y_center + radius],
                 fill=fill_color,
                 outline=outline_color,
                 width=outline_width)
    return image


MODEL_DIR = "/mmu_mllm_hdd_2/lingzhixin/release/20250613"

MODEL_DIR = "/mmu_mllm_hdd_2/zhouyang12/output1/Keye/0.8.6/Stage3/8b/v16/step1600/global_step1600/converted/"
processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
# tokenizer = processor.tokenizer


def make_inputs(a,b):
    messages = [
        # {
        #     "role": "system",
        #     "content": " ",
        # },
        {
            "role": "user",
            "content": [
                #{"type": "image", "image": generate_circle_image((a,b),) },
                #{"type": "image", "image": generate_circle_image((a,b),) },
                                {"type": "video", "video": "/mmu_mllm_hdd_2/penghao03/case_video/138571915108.mp4"},
                {"type": "text", "text": "what's in the image"},
            ],
        }
    ]


    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    # print(text); exit()
    import copy
    # image_inputs, video_inputs = process_vision_info(copy.deepcopy(messages), image_factor=None)
    image_inputs, video_inputs = process_vision_info_keye(messages, image_factor=None)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    print(inputs['pixel_values_videos'].shape)
    return messages, inputs



make_inputs(100,100)