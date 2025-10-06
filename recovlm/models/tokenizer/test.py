import torch
from recovlm.models.tokenizer.keye_tokenizer import KeyeImageTokenizer

model_path = "/llm_reco_ssd/zhouyang12/models/KeyeImageTokenizer_init/"

tokenizer = KeyeImageTokenizer.from_pretrained(model_path, _attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)

tokenizer.to("cuda:0")

x = torch.randn((16, 3, 14, 14), dtype=torch.bfloat16)
x = x.to("cuda:0")

thw = torch.LongTensor([[1, 4, 4]])
thw = thw.to("cuda:0")


output = tokenizer(x, thw)

print(output)

# print(output)