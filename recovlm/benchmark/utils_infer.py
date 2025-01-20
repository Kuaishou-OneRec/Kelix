#encodint=utf-8
from absl import flags, app

import json
import collections

from tqdm import tqdm
from vllm import LLM, SamplingParams
from ray_benchmark_dataset import * 
from torch.utils.data import DataLoader
import pandas as pd
import sys
from torch.utils.tensorboard import SummaryWriter
import re
import os
import random
import torch
import vllm
from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
import ray
import gc
import contextlib
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from ray.data import DataContext
import ast
from io import BytesIO
from PIL import Image
import numpy as np
import base64
from Levenshtein import distance

from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is"
        "The correct option is",
        "Best answer:"
        "Best option:",
        "Answer:",
        "Option:",
        "The correct answer",
        "The correct option",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCDEFGHI]", s):
        return ""
    matches = re.search(r'[ABCDEFGHI]', s)
    if matches is None:
        return ""
    return matches[0]

def get_most_similar(prediction, choices):
    """
    Use the Levenshtein distance (or edit distance) to determine which of the choices is most similar to the given prediction
    """
    distances = [distance(prediction, choice) for choice in choices]
    ind = distances.index(min(distances))
    return choices[ind]

def mathvista_extract_answer(response, question_type, answer_type, choices, precision, quick_extract=False):
    if response == "":
        return ""

    extraction = response.strip() 

    if question_type == 'multi_choice':
        # extract "A" from "(A) text"
        letter = re.findall(r'\(([a-zA-Z])\)', extraction)
        if len(letter) > 0:
            extraction = letter[0].upper()

        sequential_characters = [chr(ord('A') + i) for i in range(len(choices))]

        # if model output a character, use it as index of available choices
        if extraction in sequential_characters:
            option_index = sequential_characters.index(extraction)
            normalized_extraction = choices[option_index]
        else:
            # select the most similar option
            normalized_extraction = get_most_similar(extraction, choices)
        assert normalized_extraction in choices

    elif answer_type == 'integer':
        try:
            normalized_extraction = str(int(float(extraction)))
        except Exception:
            normalized_extraction = None

    elif answer_type == 'float':
        try:
            normalized_extraction = str(round(float(extraction), int(precision)))
        except Exception:
            normalized_extraction = None

    elif answer_type == 'list':
        try:
            normalized_extraction = str(extraction)
        except Exception:
            normalized_extraction = None

    return normalized_extraction

def get_acc(answers_dict, response_dict):
    all_count = 0
    correct = 0
    correct_keys = []
    for key, val in answers_dict.items():
        if answers_dict[key] == response_dict[key]:
            correct_keys.append(key)
            correct += 1
        all_count += 1
    return correct/all_count, correct_keys

def infer_and_eval(dataset_response, output_folder, model_step, text2index=None, is_random=False, dataset_name=""):
    if not os.path.exists(os.path.join(output_folder, model_step)):
        os.mkdir(os.path.join(output_folder, model_step))
    if not os.path.exists(os.path.join(output_folder, os.path.join(model_step, dataset_name))):
        os.mkdir(os.path.join(output_folder, os.path.join(model_step, dataset_name)))
    output_original_resp_path = os.path.join(os.path.join(output_folder, os.path.join(model_step, dataset_name)), "original_response.jsonl")
    output_answer_resp_path = os.path.join(os.path.join(output_folder, os.path.join(model_step, dataset_name)), "answer.jsonl") 
    output_original_resp = open(output_original_resp_path, "w")
    output_answer_resp = open(output_answer_resp_path, "w")
    rsp = {}
    anw = {}
    selects = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "G", "K", "L", "M"]
    pattern = r'[^a-zA-Z0-9.]*[A-M][^a-zA-Z0-9.]*'
    row = 0
    for response in dataset_response:
        key = response["ids"]
        output = response["generated_text"]
        if dataset_name in ["MMBenchEN", "MMBenchCN"]:
            anw[key] = response["answers"]
            extract = extract_characters_regex(output)
            if extract == "":
                print(f"extract is empty: {output}")
                rsp[key] = random.randint(0, 3)
            else:
                try:
                    rsp[key] = text2index[extract]
                except:
                    rsp[key] = -1
            output_original_resp.write(json.dumps({key:output}) + "\n")
            output_answer_resp.write(json.dumps({key:anw[key]}) + "\n")
        elif dataset_name in ["MMMU", "VideoMME", "MMTBench", "MMStar"]:
            anw[key] = response["answers"]
            extract = extract_characters_regex(output)
            if extract == "":
                print(f"{dataset_name} extract is empty: {output}")
                match_list = re.findall(pattern, output)
                print(f"{dataset_name} match list is {match_list}")
                if len(match_list) > 0:
                    rsp[key] = match_list[0]
                else:
                    rsp[key] = output
            else:
                rsp[key] = extract 
            output_original_resp.write(json.dumps({key:output}) + "\n")
            output_answer_resp.write(json.dumps({key:anw[key]}) + "\n")
        elif dataset_name in ["ChartQA", "OCRBench"]:
            rsp[row] = output
            anw[row] = response["answers"]
            output_original_resp.write(json.dumps({row:output}) + "\n")
            output_answer_resp.write(json.dumps({row:anw[row]}) + "\n")
            row += 1
        elif dataset_name in ["MME", "Benchmark_v21"]:
            rsp[key] = output
            anw[key] = response["answers"]
            output_original_resp.write(json.dumps({key:output}) + "\n")
            output_answer_resp.write(json.dumps({key:anw[key]}) + "\n")

        elif dataset_name in ["MathVista"]:
            rsp[key] = mathvista_extract_answer(output, response["question_type"], response["answer_type"], response["choices"], response["precision"])
            anw[key] = response["answers"]
            output_original_resp.write(json.dumps({key:output}) + "\n")
            output_answer_resp.write(json.dumps({key:anw[key]}) + "\n")

        elif dataset_name in ["Flickr30k"]:
            rsp[key] = output
            anw[key] = response["answers"]
            output_original_resp.write(json.dumps({key:output}) + "\n")
            output_answer_resp.write(json.dumps({key:anw[key]}) + "\n")

    output_original_resp.close()
    output_answer_resp.close()
    return (rsp, anw)

def dump_predict_answer(correct_keys, responses, output_path, model_path, dataset_name):
    output_error_data_path = os.path.join(os.path.join(output_path, os.path.join(model_path, dataset_name)), "predict_error_data.json")
    output_correct_data_path = os.path.join(os.path.join(output_path, os.path.join(model_path, dataset_name)), "predict_correct_data.json")
    correct_lines = []
    error_lines = []
    for i in range(len(responses)):
        cur_response = responses[i]
        cur_line = {}
        key = cur_response["ids"]
        cur_line["key"] = key
        cur_line["answer"] = cur_response["answers"]
        cur_line["messages"] = cur_response["messages"]
        cur_line["predict"] = cur_response["generated_text"]
        if key in correct_keys:
            correct_lines.append(cur_line)
        else:
            error_lines.append(cur_line)
    with open(output_error_data_path, "w") as fw:
        json.dump(error_lines, fw, indent=4, separators=(',', ':'))
    with open(output_correct_data_path, "w") as fw:
        json.dump(correct_lines, fw, indent=4, separators=(',', ':'))
    return
