from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-32B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)



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
model_inputs = tokenizer([text], return_tensors="pt")
print(model_inputs)
print("=============================")
input_text = tokenizer.decode(model_inputs["input_ids"][0], skip_special_tokens=True)
print(input_text)