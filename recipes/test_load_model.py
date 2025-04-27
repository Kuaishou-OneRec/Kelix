from recovlm.models.qwen_2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration_moonvit
from recovlm.models.qwen_2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor_moonvit 
model = Qwen2_5_VLForConditionalGeneration_moonvit.from_pretrained(
  "/llm_reco_ssd/zhouyang12/models/Qwen2-VL-7B-Instruct",allow_mismatched_sizes=True
)
model.eval()
processor = Qwen2_5_VLProcessor_moonvit.from_pretrained(
  "/llm_reco_ssd/zhouyang12/models/Qwen2-VL-7B-Instruct"
)



images = torch.randn(1, 3, 224, 224)
texts = ["hello world"]
data = processor(images=images, text=texts)
rets = model(data)
print(rets)
