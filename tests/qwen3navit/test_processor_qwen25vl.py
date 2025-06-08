from recovlm.models.qwen_3_vl_2.processing_qwen2_5_vl import Qwen2_5_VLProcessor_siglip
MODEL_DIR2="/llm_reco_ssd/zhouyang12/models/Qwen3-8B-Base-siglip"
processor = Qwen2_5_VLProcessor_siglip.from_pretrained(MODEL_DIR2)
messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Give me a short introduction to large language model."},
            ],
        }
    ]
    # Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = processor(
    text=[text],
    padding=True,
    return_tensors="pt",
)
print(inputs)
print("=============================")
inputs_text = processor.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)

print(inputs_text)
