from recovlm.models.qwen_3_vl_2.processing_qwen2_5_vl import Qwen2_5_VLProcessor_siglip
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model
import json
# load the tokenizer and the model
MODEL_DIR="/llm_reco_ssd/zhouyang12/models/Qwen3vl-8B-Base"
processor_dir = '"/llm_reco_ssd/zhouyang12/models/Qwen3-8B-Base-siglip"'
processor = Qwen2_5_VLProcessor_siglip.from_pretrained(processor_dir)
model = Qwen3Model.from_pretrained(
    MODEL_DIR,
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
        padding=True,
        return_tensors="pt",
)
inputs = inputs.to(model.device)

print('inputs', inputs)
print("input_ids",inputs.input_ids.shape)

# output = model(**inputs)

# logits = output.logits
# print(logits)


generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
#print_rank_0(output)