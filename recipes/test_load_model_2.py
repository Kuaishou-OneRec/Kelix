from transformers import AutoModelForCausalLM, AutoTokenizer
from recovlm.models.qwen_3_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor_siglip
model_name = "Qwen/Qwen3-8B"

# load the tokenizer and the model
MODEL_DIR="/llm_reco_ssd/zhouyang12/models/msy_Qwen3vl-8B-Base"
processor = Qwen2_5_VLProcessor_siglip.from_pretrained(MODEL_DIR)
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


processor = Qwen2_5_VLProcessor_siglip.from_pretrained(MODEL_DIR)
tokenizer = processor.tokenizer
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

print('model_inputs', model_inputs)

output = model(**model_inputs)

logits = output.logits
print(logits)
# Convert BFloat16 tensor to float32 before numpy conversion
logits_np = logits.detach().cpu().float().numpy().tolist()
json.dump(logits_np, open("logits2.json", "w"))