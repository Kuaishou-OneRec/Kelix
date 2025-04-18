from typing import Dict, Any, List
import io
import torch
import argparse
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


def process(args, processor, messages, images=None):
    
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

    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": "You are a helpful assistant."})

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt"
    )

    if inputs["input_ids"].shape[-1] % args.multiple_of != 0:
        padding_length = args.multiple_of - (inputs["input_ids"].shape[-1] % args.multiple_of)
        padding = inputs["input_ids"].new_full((*inputs["input_ids"][:-1], padding_length), args.pad_id)
        inputs["input_ids"] = torch.concat([inputs["input_ids"], padding], dim=-1)
    length = inputs["input_ids"].shape[-1]
    for key in inputs:
        if isinstance(inputs[key], torch.Tensor):
            inputs[key] = inputs[key].cuda()
    
    inputs["cu_seqlens"] = torch.LongTensor([0, length]).cuda()
    return inputs


def evaluate(args, model, processor):

    df = pd.read_parquet(args.file)
    tot = 0
    acc = 0
    for _, row in tqdm(df.iterrows()):
        images = parse_images(row.images)
        chosen = json.loads(row.messages) + [json.loads(row.chosen)]
        rejected = json.loads(row.messages) + [json.loads(row.rejected)]
        chosen_inputs = process(args, processor, chosen, images=images)
        rejected_inputs = process(args, processor, rejected, images=images)

        if chosen_inputs["input_ids"].shape[-1] >= 2000 or rejected_inputs["input_ids"].shape[-1] >= 2000:
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

        chosen_tokens = chosen_inputs["input_ids"].flatten()
        rejected_tokens = rejected_inputs["input_ids"].flatten()
        # assert chosen_inputs["input_ids"].flatten()[-1].item() == 198
        # assert rejected_inputs["input_ids"].flatten()[-1].item() == 198

        chosen_eos_index = (chosen_tokens != args.pad_id).nonzero().flatten()[-1].item()
        rejected_eos_index = (rejected_tokens != args.pad_id).nonzero().flatten()[-1].item()

        assert chosen_tokens[chosen_eos_index] == 198, chosen_tokens[chosen_eos_index]
        assert rejected_tokens[rejected_eos_index] == 198, rejected_tokens[rejected_eos_index]

        print(chosen_output.reward_logits.shape, rejected_output.reward_logits.shape)
        chosen_rewards = chosen_output.reward_logits.flatten().cpu()
        rejected_rewards = rejected_output.reward_logits.flatten().cpu()

        chosen_eos_reward = chosen_rewards[-1].item()
        rejected_eos_reward = rejected_rewards[-1].item()

        acc += int(chosen_eos_reward > rejected_eos_reward)
        torch.cuda.empty_cache()

    return {
        "acc": acc,
        "tot": tot
    }


def main(args):
    file = args.file
    model_dir = args.model_dir
    
    deepspeed.init_distributed()
    initialize_model_parallel(args.sp_size)

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_dir, 
        _attn_implementation="flash_attention_2",
        use_cache=False
    )
    model = model.cuda()
    model = model.to(torch.bfloat16)
    processor = Qwen2VLProcessor.from_pretrained(model_dir)
    model.eval()

    evaluate(args, model, processor)


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="/llm_reco/zangdunju/dataset/tmp/test/test.parquet")
    parser.add_argument("--model_dir", type=str, default="/llm_reco_ssd/zangdunju/models/Reward")
    parser.add_argument("--sp_size", type=int, default=1)
    parser.add_argument("--multiple_of", type=int, default=1)
    parser.add_argument("--pad_id", type=int, default=151643)
    ags = parser.parse_args()
    main(ags)
