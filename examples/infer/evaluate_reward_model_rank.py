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

from copy import deepcopy
import os.path as osp
import glob
import pandas as pd
from PIL import Image
from transformers import AutoProcessor
from recovlm.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
from recovlm.models.qwen2_vl import Qwen2VLForConditionalGeneration
from recovlm.utils.qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams
from recovlm.training.parallel import initialize_model_parallel


prompts = [
    "观看视频发表一条优质的评论内容。",
    "根据视频内容生成一条特别优质的评论。",
    "请以专业的视角分析视频中的技术细节，并生成一条优质的评论。",
    "观看视频后，写一条神评论，表达你对视频内容的感受。",
    "观看视频后，结合画面内容，写一条优质评论来描述你的感受。",
    "结合视频内容和上下文信息，为视频生成一条神评论。",
    "如果你是一个短视频观看者，在看完上述短视频之后你会有怎样的评论内容，请写出一条优质评论。",
    "请扮演一位短视频爱好者，为上述短视频生成一条优质评论内容来表达你观看后的切身体验。",
]

root = "/llm_reco/liangyiming/final_reward_data/test_data/"
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

files = glob.glob(osp.join(root, "*parquet"))
df_list = list()
for file in files:
    df = pd.read_parquet(file)
    df_list.append(df)

df = pd.concat(df_list, axis=0, ignore_index=True)
df = df.drop_duplicates(subset=["photo"])
df = df[["photo", "quality_list", "id_list", "content_list", "like_list", "reply_list", "show_cnt"]]

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
                            # print(image_block)
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
    # print(text)
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


video_dir = "/llm_reco/luoxinchen/dataset/InHouse/Photo/20250215/480p_60s_4fps_0215_0316/"
image_dir = "/llm_reco/luoxinchen/dataset/InHouse/Image/pretrain/"

tot = 0
acc = 0
rewards_list = list()
# df = df.iloc[28:]
for _, row in tqdm(df.iterrows(), total=len(df)):
    pid = str(row.photo)
    video_path = osp.join(video_dir, str(int(pid[-4:])), "{}.mp4".format(pid))
    image_path = osp.join(image_dir, pid[-4:], pid)
    image_list = glob.glob(osp.join(image_path, "*jpg"))
    if osp.exists(video_path) and osp.getsize(video_path) > 0:
        images = None
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "你是一位短视频内容理解专家。\n",
                    },
                    {
                        "type": "video",
                        "video": video_path,
                    },
                    {
                        "type": "text",
                        "text": "\n" + np.random.choice(prompts)
                    }
                ]
            }
        ]
    elif len(image_list) > 0:
        image_list.sort()
        image_basename_list = [osp.basename(x) for x in image_list]
        images = dict()
        for x in image_list:
            basename = osp.basename(x)
            image = Image.open(x)
            if image.mode != "RGB":
                image = image.convert("RGB")
            images[basename] = image
            # with open(x, "rb") as fp:
            #     image_bytes = fp.read()
            # images[basename] = base64.b64encode(image_bytes).decode("utf-8")
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "你是一位短视频内容理解专家。\n",
                    },
                    {
                        "type": "video",
                        "video": image_basename_list,
                    },
                    {
                        "type": "text",
                        "text": "\n" + np.random.choice(prompts)
                    }
                ]
            }
        ]
        # images = parse_images(json.dumps(images))
        # print(messages)
        # print(images)
    else:
        continue
    
    rewards = list()
    comments = json.loads(row.content_list)
    for i in range(min(len(comments), 200)):
        answer = comments[i]
        assistant = {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": answer
                }
            ]
        }
        chats = deepcopy(messages) + [assistant]
        
        inputs = process(processor, chats, images=images)
        if inputs["input_ids"].shape[-1] >= 3000:
            rewards.append(None)
            continue
        output = model(
            inputs["input_ids"],
            image_grid_thw=inputs.get("image_grid_thw"),
            video_grid_thw=inputs.get("video_grid_thw"),
            pixel_values=inputs.get("pixel_values"),
            pixel_values_videos=inputs.get("pixel_values_videos"),
            cu_seqlens=inputs.get("cu_seqlens")
        )
        assert inputs["input_ids"].flatten()[-1].item() == 198

        comment_reward = output.reward_logits.flatten()[-1].item()
        rewards.append(comment_reward)
    rewards_list.append(json.dumps(rewards))

    torch.cuda.empty_cache()

df["rewards"] = rewards_list
df.to_parquet("/llm_reco/zangdunju/dataset/tmp/process/tmp200.parquet", index=False)
