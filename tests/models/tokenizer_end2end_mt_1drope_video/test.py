from configuration_keye import KeyeConfig
from modeling_keye import KeyeForConditionalGeneration, KeyeImageTokenizer
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM
import random
import torch
from torch import nn

path = '/llm_reco_ssd/zhouyang12/models/KeyeImageTokenizer_end2end_init_Keye1_5_init'
tokenizer_path = '/llm_reco_ssd/zhouyang12/models/KeyeImageTokenizer_65536_codebooksize_128hid_kmeans_init'
Qwen3_path = '/llm_reco_ssd/zhouyang12/models/Keye-VL-1_5-8B-Base/'

config = KeyeConfig.from_pretrained(path)
model = KeyeForConditionalGeneration(config=config)
model_state_dict = model.state_dict()
ss = set(model_state_dict.keys())

tokenizer = KeyeImageTokenizer.from_pretrained(tokenizer_path)
tokenizer_state_dict = tokenizer.state_dict()

Qwen3 = AutoModelForCausalLM.from_pretrained(
    Qwen3_path,
    trust_remote_code=True)
keye_dict = Qwen3.state_dict()

qwen3_state_dict = {}
for k, v in keye_dict.items():
    if 'model.language_model.' in k:
        qwen3_state_dict[k] = v

print("tokenizer")
for k, v in tokenizer_state_dict.items():
    model_key = 'visual_tokenizer.' + k
    if model_key in model_state_dict:
        model_state_dict[model_key] = v
        ss.remove(model_key)
    print(k)

print("Keye")
for k, v in qwen3_state_dict.items():
    if 'model.language_model' in k:
        target_k = "model." + k.split('model.language_model.')[-1]
        if target_k in model_state_dict:
            model_state_dict[target_k] = v
            ss.remove(target_k)
            print(target_k)


print(ss, "ZDK")
# Function to verify weight mapping
def verify_weight_mapping(source_dict, target_dict, key_mapping_func=None):
    if not source_dict:
        return None, False, "Source dictionary is empty"

    # Select a random key from the source dictionary
    random_key = random.choice(list(source_dict.keys()))
    source_tensor = source_dict[random_key]

    # Apply key mapping function if provided
    target_key = key_mapping_func(random_key) if key_mapping_func else random_key

    # Check if the key exists in the target dictionary
    if target_key not in target_dict:
        return random_key, False, f"Key '{target_key}' not found in target dictionary"

    target_tensor = target_dict[target_key]

    # Check if the tensors are equal
    if torch.allclose(source_tensor, target_tensor):
        return random_key, True, f"Success: '{target_key}' in target matches '{random_key}' in source"
    else:
        return random_key, False, f"Mismatch: '{target_key}' in target does not match '{random_key}' in source"


# Verify tokenizer state dict mapping
print("\n=== Verifying Tokenizer Weight Mapping ===")
tokenizer_key, tokenizer_success, tokenizer_message = verify_weight_mapping(
    tokenizer_state_dict,
    model_state_dict,
    lambda k: f"visual_tokenizer.{k}"
)

print(f"Selected tokenizer layer: {tokenizer_key}")
print(f"Status: {'✓ Success' if tokenizer_success else '✗ Failed'}")
print(f"Message: {tokenizer_message}")

# Verify Qwen3 state dict mapping
print("\n=== Verifying Qwen3 Weight Mapping ===")
qwen3_key, qwen3_success, qwen3_message = verify_weight_mapping(
    qwen3_state_dict,
    model_state_dict,
    lambda k: "model." + k.split('model.language_model.')[-1]
)

print(f"Selected Qwen3 layer: {qwen3_key}")
print(f"Status: {'✓ Success' if qwen3_success else '✗ Failed'}")
print(f"Message: {qwen3_message}")

# Summary
print("\n=== Verification Summary ===")
if tokenizer_success and qwen3_success:
    print("✓ All selected layers were correctly mapped in model_state_dict.")
else:
    print("✗ Some layers were not co rectly mapped in model_state_dict.")

# nn.init.kaiming_normal_(model_state_dict['quant_projector.weight'], a=0, mode='fan_in', nonlinearity='relu')
#
# model.load_state_dict(model_state_dict)
# model.save_pretrained('/llm_reco_ssd/zhouyang12/models/KeyeImageTokenizer_end2end_init_Keye1_5_base_init')
