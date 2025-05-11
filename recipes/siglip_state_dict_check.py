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