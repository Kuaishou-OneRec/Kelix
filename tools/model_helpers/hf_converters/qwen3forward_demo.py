import torch 
import torch.nn as nn


def num_params(model):
    return sum([x.numel() for x in model.parameters()])


def info_params_recursive(model, name="", max_depth=5, curr_depth=0):
    """
    from torchvision import models
    print(info_params_recursive(models.resnet18()))
    """
    res = ""
    if curr_depth == 0:
        res += "下面每行的格式为:\n当前深度-<模型类型>(模型名称): 参数数量\t\tp0:第一个参数名:第一个参数均值\n"
    if curr_depth == max_depth: return ""
    #
    indent = '--' * (curr_depth + 1)
    named_params = list(model.named_parameters())
    if len(named_params):
        pname, pparam = sorted(named_params)[0]
        pparam = pparam.detach().mean().item()
    else:
        pname, pparam = None, None
    res += "{} {}-{}({}): {}\t\tp0:{}:{}\n".format(indent, curr_depth, type(model), name, num_params(model), pname, pparam)
    for name, model in model.named_children():
        if isinstance(model, nn.Module):
            res += info_params_recursive(model, name, max_depth, curr_depth + 1)
    return res

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

model_name = "/mmu_mllm_hdd_2/lingzhixin/models/Keye-32B-vit0.8.1_0606"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    _attn_implementation = 'flash_attention_2',
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)
# print(info_params_recursive( model, max_depth=5)); exit()

# prepare the model input
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
    enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    top_k=1,
    max_new_tokens=256
)
logits = model(**model_inputs).logits

output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("content:", content)
print("logits:", logits)



# def generate_circle_image(size=(200, 200), fill_color=(0, 0, 0), outline_color=(255, 255, 255), outline_width=5):
#     """
#     生成一个包含一个圆的 PIL Image 对象。

#     :param size: 图像的大小，默认为 (200, 200)
#     :param fill_color: 圆的填充颜色，默认为黑色 (0, 0, 0)
#     :param outline_color: 圆的轮廓颜色，默认为白色 (255, 255, 255)
#     :param outline_width: 圆的轮廓宽度，默认为 5
#     :return: 生成的 PIL Image 对象
#     """
#     # 创建一个新的图像对象
#     image = Image.new('RGB', size, color=(255, 255, 255))
#     draw = ImageDraw.Draw(image)
#     # 计算圆的坐标（图像中心为圆心）
#     x_center, y_center = size[0] // 2, size[1] // 2
#     radius = min(size[0], size[1]) // 2
#     # 绘制圆
#     draw.ellipse([x_center - radius, y_center - radius, x_center + radius, y_center + radius],
#                  fill=fill_color,
#                  outline=outline_color,
#                  width=outline_width)
#     return image


# print("=" * 200)

# def set_seed(seed: int):
#     import random
#     import numpy as np

#     """设置所有可能的随机数种子，保证实验可重复性"""
#     # 设置 Python 内置的随机数种子
#     random.seed(seed)
#     # 设置 NumPy 的随机数种子
#     np.random.seed(seed)
#     # 设置 PyTorch 的 CPU 随机数种子
#     torch.manual_seed(seed)
#     # 设置 PyTorch 的 CUDA 随机数种子（用于 GPU 计算）
#     torch.cuda.manual_seed(seed)
#     # 如果使用了多个 GPU，还需要设置这个
#     torch.cuda.manual_seed_all(seed)
#     # 禁用 CuDNN 的非确定性算法（确保结果可复现）
#     torch.backends.cudnn.deterministic = True
#     # 禁用 CuDNN 的自动调优功能（确保每次运行使用相同的算法）
#     torch.backends.cudnn.benchmark = False
# from PIL import Image, ImageDraw
# from recovlm.models.keye.keye_vl_utils import process_vision_info
# processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)


# # prepare the model input
# prompt = "Give me a short introduction to large language model."
# messages = [
#     {
#         "role": "user",
#         "content": [
#             {"type": "image", "image": generate_circle_image((100,100),) },
#             {"type": "text", "text": "what's in the image"},
#         ],
#     }
# ]

# text = processor.apply_chat_template(
#     messages, tokenize=False, add_generation_prompt=False
# )

# image_inputs, video_inputs = process_vision_info(messages)

# model_inputs = processor(
#     text=[text],
#     images=image_inputs,
#     videos=video_inputs,
#     padding=True,
#     return_tensors="pt",
# )

# # conduct text completion
# generated_ids = model.generate(
#     **model_inputs,
#     top_k=1,
#     max_new_tokens=256
# )
# logits = model(**model_inputs).logits

# output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# # parsing thinking content
# try:
#     # rindex finding 151668 (</think>)
#     index = len(output_ids) - output_ids[::-1].index(151668)
# except ValueError:
#     index = 0

# thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
# content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

# print("content:", content)
# print("logits:", logits)


"""
logits: tensor([[[ 4.8438,  3.0781,  3.0000,  ..., -2.5625, -2.5625, -2.5625],
         [-0.2812,  4.4375,  1.4141,  ..., -7.1562, -7.1562, -7.2812],
         [ 1.0078,  6.8438,  4.5312,  ..., -7.6875, -7.7188, -7.9375],
         ...,
         [ 4.0000,  1.7109,  5.0312,  ..., -0.4688, -0.4570, -0.3203],
         [ 6.0000,  2.6719,  2.7188,  ...,  1.0938,  1.1016,  1.0469],
         [ 1.9531,  3.8750,  5.7812,  ...,  6.1250,  6.1250,  6.0625]]],
       device='cuda:0', dtype=torch.bfloat16, grad_fn=<ToCopyBackward0>)
"""