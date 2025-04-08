from typing import Dict, Any, List
import io
import torch
import sys
sys.path.append("/llm_reco/zangdunju/vllm/rlhf/recovlm")
from tqdm import tqdm
import os
import torch.distributed as dist
import json
import base64
import collections
import deepspeed
import numpy as np

import pandas as pd
from PIL import Image
from transformers import AutoProcessor
from recovlm.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
from recovlm.models.qwen2_vl import Qwen2VLForConditionalGeneration
from recovlm.utils.qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams
from recovlm.training.parallel import initialize_model_parallel


file = "/llm_reco/zangdunju/dataset/tmp/test/test.parquet"
model_dir = "/llm_reco_ssd/zangdunju/models/Reward"
deepspeed.init_distributed()
initialize_model_parallel(1)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_dir, 
    _attn_implementation="flash_attention_2",
    use_cache=False
)
model = model.cuda()
model = model.to(torch.bfloat16)
processor = Qwen2VLProcessor.from_pretrained(model_dir)
model.eval()

df = pd.read_parquet(file)


def parse_images(images=None):
    if images is not None and len(json.loads(images)) > 0:
        images = json.loads(images)
        tmp_images = dict()
        for key in images:
            bytes = base64.b64decode(images[key])
            image = Image.open(io.BytesIO(bytes))
            if image.mode != "RGB":
                image = image.convert("RGB")
            tmp_images[key] = image
        images = tmp_images
        return images
    return None


def process(processor, messages, images=None):
    
    for message in messages:
        for block in message["content"]:
            if block["type"] == "image":
                bytes = base64.b64decode(block["image"])
                block["image"] = Image.open(io.BytesIO(bytes))
            elif block["type"] == "video":
                block["nframes"] = 10
                if isinstance(block["video"], list):
                    assert images is not None
                    for i in range(len(block["video"])):
                        image_block = block["video"][i]
                        if isinstance(image_block, str):
                            block["video"][i] = {
                                "type": "image",
                                "image": images[image_block]
                            }
                        elif isinstance(image_block, dict):
                            block["video"][i] = {
                                "type": "image",
                                "image": images[image_block["image"]]
                            }
                        else:
                            raise TypeError

    # if messages[0]["role"] != "system":
    #     messages.insert(0, {"role": "system", "content": "You are a helpful assistant."})

    # print(messages)
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    print(text)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt"
    )

    length = inputs["input_ids"].shape[-1]
    for key in inputs:
        if isinstance(inputs[key], torch.Tensor):
            inputs[key] = inputs[key].cuda()
    
    inputs["cu_seqlens"] = torch.LongTensor([0, length]).cuda()
    return inputs


tot = 0
acc = 0
for _, row in tqdm(df.iterrows()):
    images = parse_images(row.images)
    chosen = json.loads(row.messages) + [json.loads(row.chosen)]
    rejected = json.loads(row.messages) + [json.loads(row.rejected)]
    chosen_inputs = process(processor, chosen, images=images)
    rejected_inputs = process(processor, rejected, images=images)

    print(chosen_inputs["input_ids"].numel())
    if chosen_inputs["input_ids"].shape[-1] >= 3000 or rejected_inputs["input_ids"].shape[-1] >= 3000:
        continue
    tot += 1
    chosen_output = model(
        chosen_inputs["input_ids"],
        image_grid_thw=chosen_inputs.get("image_grid_thw"),
        video_grid_thw=chosen_inputs.get("video_grid_thw"),
        pixel_values=chosen_inputs.get("pixel_values"),
        pixel_values_videos=chosen_inputs.get("pixel_values_videos"),
        cu_seqlens=chosen_inputs.get("cu_seqlens")
    )
    rejected_output = model(
        rejected_inputs["input_ids"],
        image_grid_thw=rejected_inputs.get("image_grid_thw"),
        video_grid_thw=rejected_inputs.get("video_grid_thw"),
        pixel_values=rejected_inputs.get("pixel_values"),
        pixel_values_videos=rejected_inputs.get("pixel_values_videos"),
        cu_seqlens=rejected_inputs.get("cu_seqlens")
    )

    # assert chosen_inputs["input_ids"].dim() == 10, chosen_inputs["input_ids"].shape
    assert chosen_inputs["input_ids"].flatten()[-1].item() == 198
    assert rejected_inputs["input_ids"].flatten()[-1].item() == 198

    chosen_rewards = chosen_output.reward_logits.flatten()[-1].item()
    rejected_rewards = rejected_output.reward_logits.flatten()[-1].item()

    acc += int(chosen_rewards > rejected_rewards)
    # if tot == 40:
    #     break
    # print(torch.cuda.memory_allocated())
    torch.cuda.empty_cache()
    # print(torch.cuda.memory_allocated())

print(f"{acc}/{tot}")
