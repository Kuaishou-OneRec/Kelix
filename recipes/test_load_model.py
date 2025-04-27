from recovlm.models.qwen_2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration_moonvit
model = Qwen2_5_VLForConditionalGeneration_moonvit.from_pretrained(
  "/llm_reco_ssd/zhouyang12/models/Qwen2-VL-7B-Instruct"
)
model.eval()
images = torch.randn(1, 3, 224, 224)
texts = ["hello world"]
rets = model(images, texts)
print(rets)
