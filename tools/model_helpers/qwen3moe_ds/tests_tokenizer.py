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

import torch


def get_assistant_mask(batch_input_ids: torch.Tensor,
                       start_pattern,
                       end_pattern):
  if not start_pattern:
    start_pattern = [151644, 77091, 198]
  if not end_pattern:
    end_pattern = [151645, 198]

  masks = []
  for input_ids in batch_input_ids:
    mask = []
    assistant_start = []
    assistant_end = []
    to_mask = False
    for _id in input_ids:
      mask.append(int(to_mask))
      if not to_mask:
        if _id in start_pattern:
          assistant_start.append(_id.item())
        else:
          assistant_start = []
        if assistant_start[-len(start_pattern):] == start_pattern:
          to_mask = True
          assistant_start = []
      else:
        # print(324555, _id, end_pattern, _id in end_pattern)
        if _id in end_pattern:
          assistant_end.append(_id.item())
        else:
          assistant_end = []
        
        # print(assistant_end[-len(end_pattern):] == end_pattern, 23454)
        if assistant_end[-len(end_pattern):] == end_pattern:
          to_mask = False
          # print(assistant_end, 4343)
          assistant_end = []
          
    masks.append(mask)
  return torch.tensor(masks) 



'''
cp configuration_utils.py /usr/local/lib/python3.10/dist-packages/transformers/configuration_utils.py; cp processing_utils.py /usr/local/lib/python3.10/dist-packages/transformers/processing_utils.py; cp processing_auto.py /usr/local/lib/python3.10/dist-packages/transformers/./models/auto/processing_auto.py; PYTHONPATH=. python3 -m v0_8_1.Keye-2B.tests
'''

torch.cuda.set_device(0)
torch.set_default_dtype(torch.bfloat16)

current_script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(current_script_path)

MODEL_DIR = "/llm_reco/lingzhixin/models/Keye-2B-demo_dev" if len(sys.argv) == 1 else sys.argv[1]
#MODEL_DIR = "/mmu_mllm_hdd_2/zhouyang12/output1/Keye/0.8.6/Stage3/8b/v11/step400/global_step400/converted/"
MODEL_DIR1 = "/mmu_mllm_hdd_2/zhouyang12/output1/Keye/0.8.6/Stage3/8b/v11/step2800/global_step2800/converted/"
MODEL_DIR2 = "/llm_reco/lingzhixin/recovlm_qw0510/recovlm/tools/model_helpers/qwen3moe_ds/tmp"
MODEL_DIR3 = "/llm_reco/lingzhixin/recovlm_qw0510/recovlm/tools/model_helpers/qwen3moe_ds/baseR1"
# MODEL_DIR = "/mmu_mllm_hdd_2/lingzhixin/models/Keye-R1-0528-8B-vit0.8.1_0606_v0_8_1/"
# MODEL_DIR = "/mmu_mllm_hdd_2/lingzhixin/models/Keye-R1-0528-8B-vit0.8.1_0606_v0_8_1/"



# MODEL_DIR = "/mmu_mllm_hdd_2/lingzhixin/models/Keye-R1-0528-8B-vit-scratch_v0_8_1" if len(sys.argv) == 1 else sys.argv[1]
print(f"MODEL_DIR={MODEL_DIR}")

from keye_vl_utils import process_vision_info

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
    


def generate_circle_image(size=(50, 50), fill_color=(0, 0, 0), outline_color=(255, 255, 255), outline_width=5):
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


def make_inputs(a,b, model_dir):
    # https://huggingface.co/datasets/MathLLMs/MathVision, mathvision 把图片放在后面
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    # try:
    #     processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    #     fb = False
    # except:
    #     print("fall back")
    #     fb = True
    #     processor = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    # tokenizer = processor.tokenizer

    messages1 = [
        {
            "role": "system",
            "content": """
For example:
<think>
Let me analyze this question carefully...
</think>

<answer>
[Your final, concise answer here]
</answer>
"""
        },
        {
            "role": "user",
            "content": "How are you?"
        },
        {
            "role": "assistant",
            "content": "Fine, thank you.</think>斐波那契数列的第六位是5。"
        },
    ]

    # messages1 = [
    #     {
    #         "role": "user",
    #         "content":[
    #             {"type": "text", "text": "How are you?"},
    #             {"type": "image", "image": generate_circle_image()},
    #             #{"type": "video", "video": "/mmu_mllm_hdd_2/lingzhixin/recovlm_data/tests/2.mp4"},

    #         ],
    #     },
    #     {
    #         "role": "assistant",
    #         "content": [
    #             {"type": "text", "text": "Fine, thank you."},
    #         ],
    #     },斐波那契数列的定义是：第一个数是0，第二个数是1，从第三个数开始，每个数都是前两个数之和

    # ]
    # if 'R1' in model_dir:
    #     # messages = [
    #     #     {
    #     #         "role": "system",
    #     #         "content": "You are a helpful assistant."
    #     #     }
    #     # ] + messages
    #     messages1 = [
    #         {
    #             "role": "system",
    #             "content": "You are a helpful assistant."
    #         }
    #     ] + messages1
    import copy
    # messages_ = copy.deepcopy(messages)
    # text = processor.apply_chat_template(
    #     messages, tokenize=False, add_generation_prompt=False
    # )
    text1 = processor.apply_chat_template(
        messages1, tokenize=False, add_generation_prompt=False
    )
    # image_inputs, video_inputs = process_vision_info(messages)
    # inputs = processor(
    #     text=[text],
    #     #images=image_inputs,
    #     #videos=video_inputs,
    #     padding=True,
    #     return_tensors="pt",
    # )
    masks1 = get_assistant_mask(batch_input_ids=processor(text=[text1], padding=True, return_tensors="pt",)["input_ids"], 
                            start_pattern=[151670], 
                            end_pattern=[151645])
    return {
        # "messages_": messages_,
        # "text": text,
        # "inputs": inputs["input_ids"][0].tolist(),

        "masks1": masks1,
        "messages1": messages1,
        "text1": text1,
        "inputs1": processor(text=[text1], padding=True, return_tensors="pt",)
    }


# start_pattern = [151670]
# end_pattern = [151645]

# # 调用函数
# masks = get_assistant_mask(batch_input_ids=input_ids, 
#                           start_pattern=start_pattern, 
#                           end_pattern=end_pattern)


for d in [
    MODEL_DIR1, 
    MODEL_DIR2,
    #MODEL_DIR3
    ]:
    print("=" * 20)
    print(d)
    print(format_dict_or_list(
        make_inputs(100,100,d)
    ))


'''
{
  "messages_": [
    {
      "role": "user",
      "content": "How are you?"
    },
    {
      "role": "assistant",
      "content": "Fine, thank you."
    }
  ],
  "text": "<｜begin▁of▁sentence｜>You are a helpful assistant.<｜User｜>How are you?<｜Assistant｜>Fine, thank you.<｜end▁of▁sentence｜>",
  "inputs": {'input_ids': tensor([[151643,   2610,    525,    264,  10950,  17847,     13, 151669,   4340,
            525,    498,     30, 151670,  63716,     11,   9702,    498,     13,
         151645]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}
}
'''