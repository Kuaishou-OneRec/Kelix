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

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "/mmu_mllm_hdd_2/lingzhixin/models/Qwen3-32B"

from recovlm.models.qwen3 import Qwen3ForCausalLM
model_name = "/llm_reco_ssd/zhouyang12/models/Qwen3-8B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = Qwen3ForCausalLM.from_pretrained(
    model_name,
    _attn_implementation = 'flash_attention_2',
    torch_dtype="auto",
    device_map="auto"
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

"""
root@aiplatform-wlf2-ge26-18:/llm_reco/lingzhixin/recovlm_qw0510/recovlm# PYTHONPATH=. python3 tools/model_helpers/hf_converters/qwen3forward2.py 
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 17/17 [00:26<00:00,  1.58s/it]
content: 
logits: tensor([[[ 4.8438,  3.0781,  3.0000,  ..., -2.5625, -2.5625, -2.5625],
         [-0.2812,  4.4375,  1.4141,  ..., -7.1562, -7.1562, -7.2812],
         [ 1.0078,  6.8438,  4.5312,  ..., -7.6875, -7.7188, -7.9375],
         ...,
         [ 4.0000,  1.7109,  5.0312,  ..., -0.4688, -0.4570, -0.3203],
         [ 6.0000,  2.6719,  2.7188,  ...,  1.0938,  1.1016,  1.0469],
         [ 1.9531,  3.8750,  5.7812,  ...,  6.1250,  6.1250,  6.0625]]],
       device='cuda:0', dtype=torch.bfloat16, grad_fn=<ToCopyBackward0>)
"""