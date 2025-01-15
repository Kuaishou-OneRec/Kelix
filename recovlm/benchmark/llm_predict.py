#encoding=utf-8
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
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info


class LLMPredictor:

    def __init__(self, model_folder, tp, limit_mm, temperature, top_p, repetition_penalty, max_tokens):
        # Create an LLM.
        self.llm = LLM(model=model_folder,
                       tensor_parallel_size=tp,
                       gpu_memory_utilization=0.9,
                       rope_scaling={
                            "mrope_section": [
                                16,
                                24,
                                24
                            ],
                            "rope_type": "mrope",
                            "type": "mrope"
                       },
                       limit_mm_per_prompt={
                           "image": limit_mm,
                           "video": limit_mm
                           })
        
        self.processor = AutoProcessor.from_pretrained(model_folder)
        self.sampling_params = SamplingParams(
            temperature=temperature, top_p=top_p,
            repetition_penalty=repetition_penalty, max_tokens=max_tokens)

    def process(self, serialized_messages):
        data_json = json.loads(serialized_messages)
        messages = data_json["inputs"]
        for block in messages[1]["content"]:
            if block["type"] == "image":
                bytes = base64.b64decode(block["image"])
                block["image"] = Image.open(BytesIO(bytes))
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs
        inputs = {"prompt": text, "multi_modal_data": mm_data}
        data_json["inputs"] = inputs
        return data_json

    def collate(self, samples):
        batch = collections.defaultdict(list)
        for sample in samples:
            for key, item in sample.items():
                batch[key].append(item)
        return batch 

    def __call__(self, batch):
        samples = []
        for messages in batch["messages"]:
            samples.append(self.process(messages))
        batch = self.collate(samples)
        outputs = self.llm.generate(batch["inputs"], self.sampling_params)
        generated: List[str] = []
        answers = []
        ids = []
        question_types = []
        answer_types = []
        choices = []
        precisions = []
        datasets = []
        for i in range(len(outputs)):
            output = outputs[i]
            generated.append(output.outputs[0].text)
            answers.append(batch["answer"][i])
            ids.append(batch["id"][i])
            if "question_type" in batch:
                question_types.append(batch["question_type"][i])
            else:
                question_types.append("None")
            if "answer_type" in batch:
                answer_types.append(batch["answer_type"][i])
            else:
                answer_types.append("None")
            if "choices" in batch:
                choices.append(batch["choices"][i])
            else:
                choices.append("None")
            if "precision" in batch:
                precisions.append(batch["precision"][i])
            else:
                precisions.append("None")
            if "dataset" in batch:
                datasets.append(batch["dataset"])
            else:
                datasets.append("None")

        return {
            "generated_text": generated,
            "answers": answers,
            "ids": ids,
            "question_type": question_types,
            "answer_type": answer_types,
            "choices": choices,
            "precision": precisions,
            "dataset": datasets
        }