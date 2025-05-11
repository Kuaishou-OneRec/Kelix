# from transformers import AutoModelForCausalLM, AutoTokenizer

# model_name = "Qwen/Qwen3-8B"

# # load the tokenizer and the model
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype="auto",
#     device_map="auto"
# )

# # prepare the model input
# prompt = "Give me a short introduction to large language model."
# messages = [
#     {"role": "user", "content": prompt}
# ]
# text = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True,
#     enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
# )
# model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# #print the logits
# logits1 = model(**model_inputs).logits

# #torch.Size([1, 18, 151936])





from recovlm.models.qwen3_siglip.modeling_qwen3_siglip import Qwen3ForCausalLM

model = Qwen3ForCausalLM.from_pretrained('/llm_reco_ssd/zhouyang12/models/Qwen3-8B-Base')
tokenizer = AutoTokenizer.from_pretrained('/llm_reco_ssd/zhouyang12/models/Qwen3-8B-Base')

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)


# prepare the model input
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

#print the logits
logits2 = model(**model_inputs).logits

#judge the logits are the same
print(torch.allclose(logits1, logits2))


