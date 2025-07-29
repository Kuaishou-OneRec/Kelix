import sys
sys.path.append("/llm_reco/lingzhixin/recovlm_qw0510/recovlm/recovlm/models/keye")


from PIL import Image, ImageDraw
from PIL import Image
import torch
import sys
from transformers import AutoTokenizer, AutoModel, AutoProcessor
import contextlib
from transformers import AutoTokenizer, AutoModel, AutoProcessor
from keye_vl_utils import process_vision_info
import os
import shutil
import json

'''
cp configuration_utils.py /usr/local/lib/python3.10/dist-packages/transformers/configuration_utils.py; cp processing_utils.py /usr/local/lib/python3.10/dist-packages/transformers/processing_utils.py; cp processing_auto.py /usr/local/lib/python3.10/dist-packages/transformers/./models/auto/processing_auto.py; PYTHONPATH=. python3 -m v0_8_1.Keye-2B.tests
'''

torch.cuda.set_device(0)
torch.set_default_dtype(torch.bfloat16)

current_script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(current_script_path)

MODEL_DIR = "/llm_reco/lingzhixin/models/Keye-2B-demo_dev" if len(sys.argv) == 1 else sys.argv[1]
#MODEL_DIR = "/mmu_mllm_hdd_2/zhouyang12/output1/Keye/0.8.6/Stage3/8b/v11/step400/global_step400/converted/"
# MODEL_DIR = "/mmu_mllm_hdd_2/zhouyang12/output1/Keye/0.8.6/Stage3/8b/v11/step2800/global_step2800/converted/"
MODEL_DIR = "/mmu_mllm_hdd_2/zhouyang12/output1/Keye/0.8.6/Stage3/8b/v101/step2800/global_step2800/converted"
# MODEL_DIR = "/mmu_mllm_hdd_2/lingzhixin/models/Keye-R1-0528-8B-vit0.8.1_0606_v0_8_1/"
# MODEL_DIR = "/mmu_mllm_hdd_2/lingzhixin/models/Keye-R1-0528-8B-vit0.8.1_0606_v0_8_1/"
MODEL_DIR = "/mmu_mllm_hdd_2/zhouyang12/output1/Keye/0.8.6/Stage3/8b/v12/step2400/global_step2400/converted/"
MODEL_DIR = "/mmu_mllm_hdd_2/zhouyang12/output1/Keye/0.8.6/Stage3/8b/v13/step2800/global_step2800/converted/"
MODEL_DIR = "/mmu_mllm_hdd_2/zhouyang12/output1/Keye/0.8.6/Stage3/8b/v13/step400/global_step400/converted"
MODEL_DIR = "/mmu_mllm_hdd_2/lingzhixin/models/Keye-R1-0528-8B-vit0.8.1_0606-fix_v0_8_1/"
MODEL_DIR = "/mmu_mllm_hdd_2/zhouyang12/output1/Keye/0.8.6/Stage3/8b/v15/step10/global_step10/converted"
MODEL_DIR = "/mmu_mllm_hdd_2/zhouyang12/output1/Keye/0.8.6/Stage3/8b/v15/step2800/global_step2800/converted/"

# MODEL_DIR = "/mmu_mllm_hdd_2/lingzhixin/models/Keye-R1-0528-8B-vit-scratch_v0_8_1" if len(sys.argv) == 1 else sys.argv[1]
print(f"MODEL_DIR={MODEL_DIR}")

from keye_vl_utils import process_vision_info
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
local_rank = 0


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




def make_inputs(a,b, _type="image", with_sys=True):
    video = {"type": "video", "video": "/llm_reco/lingzhixin/recovlm_data/tests/2.mp4", "max_pixels": 256*28*28, "nframes": 4}
    image = {"type": "image", "image": generate_circle_image((a,b),) , "max_pixels": 256*28*28}

    mm = [{"type": "image", "image": generate_circle_image((a,b),) }] if _type == "image" else [video]
    if _type is None: mm = []
    if _type == "both":
        mm = [
            video, # 这里是demo, 评估可以用更大的max_pixels
            image,
        ]

    messages = [
        {
            "role": "system",
            "content": """                            
You are a AI assistant expert in reasoning. 
Before answering any question, 
you should first think step-by-step, 
then provide your conclusion based on your reasoning process. 
Your output should follow this format:

First, include your thinking process within <think></think> tags.
Then, provide your final answer within <answer></answer> tags.

For example:
<think>
Let me analyze this question carefully...
</think>

<answer>
[Your final, concise answer here]
</answer>
""" if with_sys else "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": [
                *mm,
                {"type": "text", "text": "what's in the image"} if len(mm) else {"type": "text", "text": "斐波那契数列的第六位是什么？"},
            ],
        }
    ]
    # if not with_sys:
    #     messages = messages[1:]
    import copy
    messages_ = copy.deepcopy(messages)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(local_rank)
    # print(format_dict_or_list(messages)); exit()
    return messages, inputs, messages_




def generate_ascii_banner(text, width=None, padding=50, char='='):
    if width is None:
        width = len(text) + 2 * padding
    
    min_width = len(text) + 4
    if width < min_width:
        width = min_width
    
    total_decor = width - len(text)
    left_decor = total_decor // 2
    right_decor = total_decor - left_decor
    
    decoration = char * width
    content_line = f"{char * left_decor}{text}{char * right_decor}"
    
    return f"{decoration}\n{content_line}\n{decoration}"
    
MODEL_DIR = "/mmu_mllm_hdd_2/zhouyang12/output1/Keye/0.8.6/Stage3/8b/v16/step2400/global_step2400/converted/"
MODEL_DIR = "/mmu_mllm_hdd_2/zhouyang12/output1/Keye/0.8.6/Stage3/8b/v17/step2400/global_step2400/converted/"

for MODEL_DIR in [
        # "/mmu_mllm_hdd_2/zhouyang12/output1/Keye/0.8.6/Stage3/8b/v12/step2400/global_step2400/converted/",
        "/mmu_mllm_hdd_2/zhouyang12/output1/Keye/0.8.6/Stage3/8b/v116/step2400/global_step2400/converted/",
        "/mmu_mllm_hdd_2/zhouyang12/output1/Keye/0.8.6/Stage3/8b/v117/step2400/global_step2400/converted/"
    ]:
    print(f"\n\nMODEL_DIR={MODEL_DIR}")
    processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
    tokenizer = processor.tokenizer
    model = AutoModel.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
        _attn_implementation = 'flash_attention_2',
        device_map="cuda:0",
        # ignore_mismatched_sizes=True,
        trust_remote_code=True
    )
    for _type in [None, "image"]:
        for with_sys in [True, False]:
            print(generate_ascii_banner(f"testing " + str(_type) +  f"with_sys={with_sys} ..." ))
            messages, inputs, messages_old = make_inputs(100,100, _type, with_sys=with_sys)
            generated = model.generate(**inputs, max_new_tokens=1024) 
            logits = model(**inputs).logits
            output_ids = generated[0][len(inputs.input_ids[0]):].tolist() 
            content = tokenizer.decode(output_ids[0:], skip_special_tokens=True).strip("\n")
            messages = messages[0]

            messages["input_ids"] = inputs.input_ids.shape
            messages["pixel_values"] = inputs.pixel_values.shape if hasattr(inputs, "pixel_values") else None
            messages["pixel_values_videos"] = inputs.pixel_values_videos.shape if hasattr(inputs, "pixel_values_videos") else None

            messages["messages"] = messages_old
            messages["content"] = content
            messages["logits"] = logits # [-1, -3:]
            messages["logits.shape"] = logits.shape
            print(format_dict_or_list(messages))


