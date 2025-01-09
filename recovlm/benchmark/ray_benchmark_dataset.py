"""I2I Pairwise Dataset"""
import numpy as np
import collections

from torch.utils.data import DataLoader
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import os

import json
from torch.utils.data import IterableDataset, DataLoader
from io import BytesIO
import pandas as pd
from PIL import Image
import ast

def MMMU_parse(sample):
    if sample["question_type"] == "multiple-choice":
        # multi choice
        prompt_1 = "Select the best answer to the following multiple-choice question based on the above images. Respond with only the letter ({0}) of the correct option. The question is "
        prompt_3 = "The best answer is: "
        prompt_2 = []

        options = ast.literal_eval(sample["options"])
        selects = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "G", "K", "L", "M"]
        selects_options = []
        if len(options) == 0:
            print("error options: ", sample["options"], sample)
        for i in range(len(options)):
            selects_options.append(selects[i])
            prompt_2.append(selects[i] + ". " + options[i])

        prompt_1 = prompt_1.format(",".join(selects_options))
        if sample["explanation"] is not None:
            content_text ="\n".join([prompt_1] + [sample["question"]] + ["The explanation of the question is " + sample["explanation"]] + ["The options are "] + prompt_2 + [prompt_3])
        else:
            content_text ="\n".join([prompt_1] + [sample["question"]] + ["The options are "] + prompt_2 + [prompt_3])
    else:
        prompt_1 = "Based on the above images, answer the following question. The question is "
        if sample["explanation"] is not None:
            content_text = "\n".join([prompt_1, sample["question"], "The explanation of the question is " + sample["explanation"], "The answer is"])
        options = [] 
    # get content
    content_image = []
    for i in range(1, 7):
        if sample[f"image_{i}"] == None:
            continue
        content_image.append({
            "type": "text",
            "text": f"This is <image {i}>."
        })
        content_image.append({
            "type": "image",
            "image": base64.b64encode(sample[f"image_{i}"]["bytes"]).decode("utf-8")
        })

    messages =[ 
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": content_image + [
                {"type": "text", "text": content_text},
            ],
        }]

    return {"messages": json.dumps({"id": sample["id"], "answer": sample["answer"], "inputs": messages})}


def VideoMME_parse(sample) -> dict:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "max_frames": 128,
                    #"resized_height": 336,
                    #"resized_width": 336,
                    "video": os.path.join("/mmu_mllm_hdd/dinghaojie/QA/Benchmarks/Video-MME/zips/data", f"{sample['videoID']}.mp4")
                },
                {"type": "text", "text": "\n".join(sample["question"].split("\n")[1:])},
            ],
        },
    ] 
    return {"messages": json.dumps({"id": sample["videoID"] + "_" + sample["question_id"], "answer": sample["answer"], "inputs": messages})}

def ChartQA_parse(sample) -> dict:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": base64.b64encode(open(sample["image_path"], "rb").read()).decode("utf-8")
                },
                {"type": "text", "text": sample["question"]},
            ],
        },
    ]

    return {"messages": json.dumps({"id": sample["key"], "answer": sample["answer"], "inputs": messages})}

def MME_parse(sample) -> dict:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": base64.b64encode(open(sample["image_path"], "rb").read()).decode("utf-8")
                },
                {"type": "text", "text": sample["question"]},
            ],
        },
    ]

    return {"messages": json.dumps({"id": sample["key"], "answer": sample["answer"], "inputs": messages, "category": sample["category"]})}

