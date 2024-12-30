import webdataset as wds
from PIL import Image
from tqdm import tqdm
import wids
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from qwen_vl_utils import process_vision_info

from transformers import AutoProcessor

# 创建 widsindex
# cd /llm_reco_ssd/luoxinchen/dataset/cc12m
# widsindex create cc12m/*.tar --output cc12m-index.json

transform_train = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406],
            std=[
                0.229,
                0.224,
                0.225]),
    ])

processor = AutoProcessor.from_pretrained(
    "/llm_reco_ssd/zhouyang12/models/Qwen2-VL-7B-Instruct")


def make_sample(sample):
  image = sample[".jpg"]
  label = sample[".txt"]
  messages = [
      {
          "role": "user",
          "content": [
              {
                  "type": "image",
                  "image": image,
              },
              {"type": "text", "text": "Describe this image."},
          ],
      }
  ]
  if image.mode != "RGB":
    image = image.convert("RGB")
  text = processor.apply_chat_template(
      messages, tokenize=False, add_generation_prompt=True)
  image_inputs, video_inputs = process_vision_info(messages)
  inputs = processor(
      text=[text],
      images=image_inputs,
      videos=video_inputs,
      padding=True,
      return_tensors="pt",
  )
  print(processor.tokenizer.decode(inputs["input_ids"][0]))
  return inputs, label

# dataset = wds.WebDataset("/llm_reco_ssd/luoxinchen/dataset/cc12m/cc12m/00001.tar").shuffle(1000).decode("pil").to_tuple("jpg;png", "json")


dataset = wids.ShardListDataset(
    "/llm_reco_ssd/luoxinchen/dataset/cc12m/cc12m-index.json")
dataset.add_transform(make_sample)

trainsampler = wids.DistributedChunkedSampler(
    dataset, chunksize=1000, shuffle=True)

trainloader = DataLoader(
    dataset,
    batch_size=16,
    num_workers=4,
    sampler=trainsampler)

for s in tqdm(dataset):
  print(s)
  break
