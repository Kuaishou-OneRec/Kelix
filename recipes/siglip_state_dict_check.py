from transformers import AutoModelForCausalLM, AutoTokenizer
from recovlm.models.qwen_3_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor_siglip
model_name = "Qwen/Qwen3-8B"
import json
# load the tokenizer and the model
processor = Qwen2_5_VLProcessor_siglip.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
for name, param in model.named_parameters():
    print(name, param.shape)