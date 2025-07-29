from transformers import AutoModelForCausalLM, AutoTokenizer
# pip3 install transformers==4.52
# torchao 0.7.0+cu118 -> 0.11.0
# /usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py


model_name = "Qwen/Qwen3-8B"
model_name = "/mmu_mllm_hdd_2/lingzhixin/models/Qwen3-32B-A3B"
# load the tokenizer and the model

import torch
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    _attn_implementation = 'flash_attention_2',
    device_map="cuda:0",
).cuda()

# prepare the model input
prompt = "Give me a short introduction to large language models."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
    enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
)
# '<|im_start|>user\nGive me a short introduction to large language models.<|im_end|>\n<|im_start|>assistant\n', enable_thinking=True, add_generation_prompt=True
# '<|im_start|>user\nGive me a short introduction to large language models.<|im_end|>\n'

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=16
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# the result will begin with thinking content in <think></think> tags, followed by the actual response
print(tokenizer.decode(output_ids, skip_special_tokens=True))