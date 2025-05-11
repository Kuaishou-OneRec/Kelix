from transformers import AutoModelForCausalLM, AutoTokenizer
from recovlm.models.qwen_3_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor_siglip
model_name = "Qwen/Qwen3-8B"
import json
# load the tokenizer and the model
MODEL_DIR="/llm_reco_ssd/zhouyang12/models/msy_Qwen3vl-8B-Base"
processor = Qwen2_5_VLProcessor_siglip.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Give me a short introduction to large language model."},
        ],
    }
]
text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
)
inputs = processor(
        text=[text],
        # images=image_inputs,
        # videos=video_inputs,
        padding=True,
        return_tensors="pt",
)
inputs = inputs.to(model.device)

print('inputs', inputs)
print("input_ids",inputs.input_ids.shape)

output = model(**inputs)

logits = output.logits
print(logits)
# Convert BFloat16 tensor to float32 before numpy conversion
logits_np = logits.detach().cpu().float().numpy().tolist()
json.dump(logits_np, open("logits2.json", "w"))