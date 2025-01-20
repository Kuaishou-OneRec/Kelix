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
import base64

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
    print(f"answer is {sample['answer']}")

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

def MMTBench_parse(sample) -> dict:
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

    return {"messages": json.dumps({"id": sample["index"], "answer": sample["answer"], "inputs": messages})}

def MMStar_parse(sample) -> dict:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": base64.b64encode(open(sample["meta_info"]["image_path"], "rb").read()).decode("utf-8")
                },
                {"type": "text", "text": "Select the best answer to the following multiple-choice question based on the above image. Question: " + sample["question"] + "\nPlease select the correct answer from the options above. \nAnswer with the option's letter from the given choices directly, such as answer letter 'A' only."},
            ],
        },
    ]

    return {"messages": json.dumps({"id": sample["index"], "answer": sample["answer"], "inputs": messages})}

def MathVista_parse(sample) -> dict:
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

    if sample["choices"] != None:
       # choices = ast.literal_eval(sample["choices"])
        choices = sample["choices"]
    else:
        choices = None

    return {"messages": json.dumps({"id": sample["pid"], 
            "answer": sample["answer"], 
            "inputs": messages, 
            "question_type": sample["question_type"], 
            "answer_type": sample["answer_type"], 
            "choices": choices, 
            "precision": sample["precision"]})}

def MMBench_parse(sample) -> dict:
    index = sample['index']
    image = sample['image']['bytes']
    answer = sample['answer']
    hint = sample['hint'] if sample['hint'] else 'N/A'
    question = sample['question']
    multiple_choices = ['A', 'B', 'C', 'D']
    prompt = 'Select the best answer to the following multiple-choice question based on the above images. Respond with only the letter (A, B, C or D) of the correct option. Context: {}\nQuestion: {}\nOptions: {}\nAnswer:'

    choice_list = []
    for i, c in enumerate(multiple_choices):
        choice_list.append('{}. {}'.format(multiple_choices[i], sample[multiple_choices[i]]))
    choice_txt = '\n'.join(choice_list)

    prompt = prompt.format(hint, question, choice_txt)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": "<img>"
                },
                {
                    "type": "image",
                    "image": base64.b64encode(image).decode("utf-8"),
                },
                {
                    "type": "text", 
                    "text": "</img>"
                },
                {
                    "type": "text", 
                    "text": prompt
                },
            ],
        },
    ]
    return {"messages": json.dumps({"id": index, "answer": answer, "inputs": messages})} 

def OCRBench_parse(sample) -> dict:
    image = sample['image']['bytes']
    answer = sample['answer']
    question = sample['question']
    question_type = sample['question_type']
    dataset = sample['dataset']
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": base64.b64encode(image).decode("utf-8"),
                },
                {
                    "type": "text", 
                    "text": "Please answer the question according to the above image, question is: " + question
                },
            ],
        },
    ]
    return {"messages": json.dumps({"id": "OCRBench", "answer": answer, "inputs": messages, "question_type": question_type, "dataset": dataset, "image": base64.b64encode(image).decode("utf-8")})} 

# def Flickr30k_parse(sample) -> dict:
#     image = "/llm_reco_ssd/luoxinchen/RecoVLM/Benchmark/dataset/flickr30k/flickr30k-images/" + sample["filename"]
#     messages = [
#         {"role": "system", "content": "You are a helpful assistant."},
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "image",
#                     "image": base64.b64encode(open(image, "rb").read()).decode("utf-8"),
#                 },
#                 {
#                     "type": "text", 
#                     "text": "Caption the above image using one simple sentence."
#                 },
#             ],
#         },
#     ]
#     return {"messages": json.dumps({"id": sample["img_id"], "answer": sample["raw"], "inputs": messages})} 

def Flickr30k_parse(sample) -> dict:
    image_name = sample['image'].split('/')[-1]
    image = "/llm_reco_ssd/luoxinchen/RecoVLM/Benchmark/dataset/flickr30k/flickr30k-images/" + image_name
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": base64.b64encode(open(image, "rb").read()).decode("utf-8"),
                },
                {
                    "type": "text", 
                    "text": "Caption the above image using one simple sentence."
                },
            ],
        },
    ]
    return {"messages": json.dumps({"id": sample["image_id"], "answer": sample["caption"], "inputs": messages})} 

def Benchmark_v21_parse(sample) -> dict:
    prompt = '''根据以上图片，回答下述问题，问题： {}'''.format(sample["question"])
    messages = [
        {"role": "system", "content": "你是一个有帮助且精准的助手."},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": base64.b64encode(open(sample["image_path"], "rb").read()).decode("utf-8")
                },
                {"type": "text", "text": prompt},
            ],
        },
    ]

    return {"messages": json.dumps({"id": sample["key"], 
            "answer": sample["answer"], 
            "inputs": messages, 
            "question_type": sample["question_type"], 
            "task_type": sample["task_type"], 
            "question": sample["question"]})}
