from recovlm.utils.ds_utils import format_dict_or_list
from PIL import Image, ImageDraw
from recovlm.training.common import set_default_dtype, get_global_grad_norm, clip_grad_by_value
from recovlm.utils.qwen_vl_utils import process_vision_info
from PIL import Image
import torch
from recovlm.training.common import set_default_dtype
from recovlm.models.keye.modeling_keye import KeyeForConditionalGeneration
from recovlm.models.keye.processing_keye import KeyeProcessor


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


MODEL_DIR = "/llm_reco_ssd/zhouyang12/models/Keye-8B-demo/"
processor = KeyeProcessor.from_pretrained(MODEL_DIR)
tokenizer = processor.tokenizer


def make_inputs(a,b):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": generate_circle_image((a,b),) },
                {"type": "text", "text": "what's in the image"},
            ],
        }
    ]


    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    image_inputs, video_inputs = process_vision_info(messages, image_factor=None)
    print(image_inputs, video_inputs)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return messages, inputs


logits_all = []
if 1:
    try:
        with set_default_dtype(torch.bfloat16):
            model = KeyeForConditionalGeneration.from_pretrained(
                MODEL_DIR,
                torch_dtype=torch.bfloat16,
                _attn_implementation = 'flash_attention_2',
                device_map="cuda:0",
                ignore_mismatched_sizes=True
            )


            messages, inputs = make_inputs(100,100)
            for k in inputs: inputs[k] = inputs[k].cuda()

            generated = model.generate(**inputs, max_new_tokens=32768)
            logits = model(**inputs).logits
            output_ids = generated[0][len(inputs.input_ids[0]):].tolist() 
            content = tokenizer.decode(output_ids[0:], skip_special_tokens=True).strip("\n")

            messages = messages[0]
            messages["content"] = content
            messages["logits"] = logits
            print(format_dict_or_list(messages))

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(e)
        pass

