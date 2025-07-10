from PIL import Image, ImageDraw
from PIL import Image
import torch
import sys
  
sys.path.append("./recovlm/models")
from keye_vitrope_slowfast.modeling_keye import KeyeForConditionalGeneration
from keye_vitrope_slowfast.processing_keye import KeyeProcessor
from keye_vitrope_slowfast.keye_vl_utils import process_vision_info
from torch.nn import CrossEntropyLoss



import contextlib

@contextlib.contextmanager
def set_default_dtype(dtype: torch.dtype):
    """
    Context manager to set torch's default dtype.

    Args:
        dtype (torch.dtype): The desired default dtype inside the context manager.

    Returns:
        ContextManager: context manager for setting default dtype.

    Example:
        >>> with set_default_dtype(torch.bfloat16):
        >>>     x = torch.tensor([1, 2, 3])
        >>>     x.dtype
        torch.bfloat16

    """
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


def format_dict_or_list(obj, indent_level=0, indent_size=2):
    """
    格式化打印dict/list，用来替代json.dumps
    """
    def format_value(value, indent_level=0, indent_size=2):
        if isinstance(value, (dict, list)):
            return format_dict_or_list(value, indent_level, indent_size)
        elif isinstance(value, str):
            return f'"{value}"'
        else:
            return str(value)

    if isinstance(obj, dict):
        items = [f": {format_value(v, indent_level + 1)}" for k, v in obj.items()]
        keys = [f'"{k}"' for k in obj.keys()]
        formatted_items = ',\n'.join(f'{(" " * indent_size * (indent_level + 1))}{k}{v}' for k, v in zip(keys, items))
        return '{\n' + formatted_items + '\n' + (' ' * indent_size * indent_level) + '}'
    elif isinstance(obj, list):
        items = [format_value(item, indent_level + 1) for item in obj]
        formatted_items = ',\n'.join(' ' * indent_size * (indent_level + 1) + item for item in items)
        return '[\n' + formatted_items + '\n' + (' ' * indent_size * indent_level) + ']'
    else:
        return obj
    

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


PROCESSOR_DIR = "/llm_reco_ssd/zhouyang12/models/Keye-8B-demo_hf_vit_rope_slowfast_0703_patch14"
MODEL_DIR = "/llm_reco_ssd/zhouyang12/models/Keye-8B-demo_hf_vit_rope_slowfast_0703_patch14"

processor = KeyeProcessor.from_pretrained(PROCESSOR_DIR, use_fast=True, local_files_only=False)
tokenizer = processor.tokenizer

def make_inputs(a,b):
    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "image", "image": generate_circle_image((a,b),) },
    #             {"type": "text", "text": "what's in the image"},
    #         ],
    #     }
    # ]

    # messages = [
    #     {"role": "user", 
    #     "content": 
    #     [
    #         {"type": "text", 
    #         "text": "\\nGenerate video descriptions that include details of visual effects, character actions, and movement of people/objects within frames. Describe this video and its style to generate a description. Pay attention to all objects in the video. Do not describe each frame individually. Instead of describing the imaginary content, only describing the content one can determine confidently. Do not describe the contents by itemizing them in list form."}]}
    #     ]

    # messages = [
    #     {"role": "user", 
    #     "content": 
    #     [
    #         {"type": "video", 
    #         "video": "/llm_reco/maosiyang/dataset/cinepile/mp4/DcZgodKg9OQ.mp4"}, 
    #         {"type": "text", 
    #         "text": "\\nGenerate video descriptions that include details of visual effects, character actions, and movement of people/objects within frames. Describe this video and its style to generate a description. Pay attention to all objects in the video. Do not describe each frame individually. Instead of describing the imaginary content, only describing the content one can determine confidently. Do not describe the contents by itemizing them in list form."}]}
    #     ]
    

    messages = [
        {"role": "user", 
        "content": 
        [
            # {"type": "video", 
            # "video": "/llm_reco_ssd/caojiangxia/vllm/sample_videos/SampleVideo_1280x720_30mb.mp4"}, 
            # {"type": "video", 
            # "video": "/llm_reco_ssd/caojiangxia/vllm/sample_videos/SampleVideo_1280x720_20mb.mp4"}, 
            # {"type": "video", 
            # "video": "/llm_reco_ssd/caojiangxia/vllm/sample_videos/SampleVideo_1280x720_10mb.mp4"}, 
            # {"type": "video", 
            # "video": "/llm_reco_ssd/caojiangxia/vllm/sample_videos/SampleVideo_1280x720_5mb.mp4"}, 
            # {"type": "video", 
            # "video": "/llm_reco_ssd/caojiangxia/vllm/sample_videos/SampleVideo_1280x720_2mb.mp4"}, 
            # {"type": "video", 
            # "video": "/llm_reco_ssd/caojiangxia/vllm/sample_videos/SampleVideo_1280x720_1mb.mp4"}, 
            {"type": "video", 
            "video": ["/llm_reco_ssd/caojiangxia/vllm/test_image.png", "/llm_reco_ssd/caojiangxia/vllm/test_image.png"] * 30}, 
            # {"type": "image", 
            # "image": "/llm_reco_ssd/caojiangxia/vllm/test_wangxiangu.png"},
            # {"type": "image", 
            # "image": "/llm_reco_ssd/caojiangxia/vllm/test_wangxiangu.png"}, 
            {"type": "text", 
            "text": "\\What is this?"}]},
        {"role": "assistant", 
        "content": 
        [
            {"type": "text", 
            "text": "\\This is a movie, "}]},
        ]

    # messages = [
    #     {"role": "user", 
    #     "content": 
    #     [
    #         {"type": "video", 
    #         "video": ["/llm_reco_ssd/caojiangxia/vllm/test_wangxiangu.png", "/llm_reco_ssd/caojiangxia/vllm/test_wangxiangu.png"]}, 
    #         {"type": "video", 
    #         "video": ["/llm_reco_ssd/caojiangxia/vllm/test_wangxiangu.png", "/llm_reco_ssd/caojiangxia/vllm/test_wangxiangu.png"]}, 
    #         # {"type": "video", 
    #         # "video": "/llm_reco/lingzhixin/recovlm_qw0510/recovlm/tests/qwen3navit/2.mp4"}, 
    #         {"type": "text", 
    #         "text": "\\nWhat information in this video?"}]}
    #     ]

    # messages = [
    #     {"role": "user", 
    #     "content": 
    #     [
    #         {"type": "image", 
    #         "image": "/llm_reco_ssd/caojiangxia/vllm/test_wangxiangu.png"}, 
    #         {"type": "text", 
    #         "text": "\\nWhat is this?"}]}
    #     ]


    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    image_inputs, video_inputs = process_vision_info(messages)
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
        # with set_default_dtype(torch.bfloat16):
            model = KeyeForConditionalGeneration.from_pretrained(
                MODEL_DIR,
                torch_dtype=torch.bfloat16,
                _attn_implementation = 'flash_attention_2',
                device_map="cuda:0",
                ignore_mismatched_sizes=True,
            )

            # model.load_state_dict(torch.load("/llm_reco_ssd/zhouyang12/models/Keye-2B-demo_fix_improc_slowfast/pytorch_model.bin"), assign=True, strict = False)

            messages, inputs = make_inputs(200,200)
            for k in inputs: inputs[k] = inputs[k].cuda()
            

            tokenized = tokenizer.apply_chat_template(
                [messages],
                # chat_template=chat_template,
                return_assistant_tokens_mask=True,
                return_dict=True
            )

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

