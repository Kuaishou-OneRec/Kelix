"""Batch inference for Qwen2-VL"""
from absl import flags, app

import json
import collections

from tqdm import tqdm
from vllm import LLM, SamplingParams
from ray_benchmark_dataset import VideoMME_parse, MMMU_parse  
from torch.utils.data import DataLoader
import pandas as pd
import sys
sys.path.insert(0, 'eval/MMMU/mmmu')
sys.path.insert(0, 'eval/MMBench')
from eval.MMMU.mmmu.main_eval_only import MainEvalOnly
from eval.MMBench.mmbench_evaluation_tricky import MMBenchEvaluation
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

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "model_folder", None, "The folder of model."
)

flags.DEFINE_string(
    "model_name", None, "The name of model."
)

flags.DEFINE_float(
    "top_p", 0.8, "The top_p params"
)

flags.DEFINE_float(
    "temperature", 0.7, "The temperature params."
)

flags.DEFINE_integer(
    "max_tokens", 512, "The max tokens to generate."
)

flags.DEFINE_integer(
    "tp", 4, "The tensor_parallel_size"
)

flags.DEFINE_string(
    "MMMU_path", None, "MMMU dataset path"
)

flags.DEFINE_string(
    "VideoMME_path", None, "VideoMME dataset path"
)

flags.DEFINE_string(
    "MMBenchEN_path", None, "MMBenchEN dataset path"
)

flags.DEFINE_string(
    "output_path", None, "The path of file to write results." 
)

flags.DEFINE_integer(
    "limit_mm_per_prompt", 10, "The maximum images and videos of mm_input per prompt"
)

flags.DEFINE_integer(
    "batch_size", 10, "The batch size for inference."
)

flags.DEFINE_float(
    "repetition_penalty", 1.05, "The maximum images of mm_input per prompt"
)

flags.DEFINE_string(
    "benchmark_original_data", None, "original benchmark data"
)
flags.DEFINE_string(
    "logging_folder", "log", "logging folder"
)
flags.DEFINE_string(
    "infer_chekpoint_file", "", "save infer checkpoint file"
)
def collate_fn(samples):
    batch = collections.defaultdict(list)
    for sample in samples:
        for key, item in sample.items():
            batch[key].append(item)
    return batch

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

    if len(s.split()) > 10 and not re.search("[ABCD]", s):
        return ""
    matches = re.search(r'[ABCD]', s)
    if matches is None:
        return ""
    return matches[0]

def get_acc(answers_dict, response_dict):
    all_count = 0
    correct = 0
    for key, val in answers_dict.items():
        if answers_dict[key] == response_dict[key]:
            correct += 1
        all_count += 1
    return correct/all_count

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
    pattern = r'[^a-zA-Z0-9.]*[A-Z][^a-zA-Z0-9.]*'
    for response in dataset_response:
        key = response["ids"]
        anw[key] = response["answers"]
        output = response["generated_text"]
        if dataset_name in ["MMBenchEN"]:
            extract = extract_characters_regex(output)
            if extract == "":
                print(f"extract is empty: {output}")
                rsp[key] = random.randint(0, 3)
            else:
                rsp[key] = text2index[extract]
            output_original_resp.write(json.dumps({key:output}) + "\n")
            output_answer_resp.write(json.dumps({key:rsp[key]}) + "\n")
        elif dataset_name in ["MMMU", "VideoMME"]:
            extract = extract_characters_regex(output)
            if extract == "":
                print(f"extract is empty: {output}")
                match_list = re.findall(pattern, output)
                print(f"match list is {match_list}")
                if len(match_list) > 0:
                    rsp[key] = match_list[0]
                else:
                    rsp[key] = output
            else:
                rsp[key] = extract 
            output_original_resp.write(json.dumps({key:output}) + "\n")
            output_answer_resp.write(json.dumps({key:rsp[key]}) + "\n")
        elif dataset_name in ["ChartQA"]:
            rsp[key] = output
            output_original_resp.write(json.dumps({key:output}) + "\n")
            output_answer_resp.write(json.dumps({key:rsp[key]}) + "\n")

    output_original_resp.close()
    output_answer_resp.close()
    return (rsp, anw)


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
        for i in range(len(outputs)):
            output = outputs[i]
            generated.append(output.outputs[0].text)
            answers.append(batch["answer"][i])
            ids.append(batch["id"][i])
        return {
            "generated_text": generated,
            "answers": answers,
            "ids": ids
        }

def scheduling_strategy_fn():
    # One bundle per tensor parallel worker
    pg = ray.util.placement_group(
        [{
            "GPU": 1,
            "CPU": 1
        }] * FLAGS.tp,
        strategy="STRICT_PACK",
    )
    return dict(scheduling_strategy=PlacementGroupSchedulingStrategy(
        pg, placement_group_capture_child_tasks=True))

def main(_):
    if not os.path.exists(FLAGS.logging_folder):
        os.mkdir(FLAGS.logging_folder)
    writer = SummaryWriter(FLAGS.logging_folder)
    model_folder = FLAGS.model_folder
    model_name = FLAGS.model_name
    last_steps = [] 
    if os.path.exists(FLAGS.infer_chekpoint_file):
        fr = open(FLAGS.infer_chekpoint_file, "r")
        for line in fr.readlines():
            last_steps.append(int(line.strip()))
    print(f"last step is {last_steps}")

    resources_kwarg: Dict[str, Any] = {}
    if FLAGS.tp == 1:
        # For tensor_parallel_size == 1, we simply set num_gpus=1.
        resources_kwarg["num_gpus"] = 1
    else:
        # Otherwise, we have to set num_gpus=0 and provide
        # a function that will create a placement group for
        # each instance.
        resources_kwarg["num_gpus"] = 0
        resources_kwarg["ray_remote_args_fn"] = scheduling_strategy_fn

    fn_constructor_kwargs = {
        "model_folder": "vllm_model_ray", 
        "tp": FLAGS.tp, 
        "limit_mm": FLAGS.limit_mm_per_prompt, 
        "temperature": FLAGS.temperature, 
        "top_p": FLAGS.top_p, 
        "repetition_penalty": FLAGS.repetition_penalty, 
        "max_tokens": FLAGS.max_tokens
    }
    
    DataContext.get_current().verbose_stats_logs = True
    DataContext.get_current().log_internal_stack_trace_to_stdout = True
    ctx = ray.data.DataContext.get_current()
    ctx.verbose_stats_logs = True
    ctx.log_internal_stack_trace_to_stdout = True
    ctx.execution_options.resource_limits.cpu = 10
    # ctx.execution_options.resource_limits.gpu = 5
    ctx.execution_options.resource_limits.object_store_memory = 20e9
    ctx.actor_task_retry_on_errors = True

    #MMMU
    # data_files = []
    # for subfolder in os.listdir(FLAGS.MMMU_path):
    #     for subfile in os.listdir(os.path.join(FLAGS.MMMU_path, subfolder)):
    #         if subfile.startswith("validation"):
    #             file_name = os.path.join(FLAGS.MMMU_path, subfolder, subfile)
    #             data_files.append("local://" + file_name)
    # print(f"data files: {data_files}")
    # MMMU_dataset = ray.data.read_parquet(data_files).map(MMMU_parse)

    #VideoMME
    with open(FLAGS.VideoMME_path, 'r') as file_:
            videoMME_data = json.load(file_)["annotations"]
    VideoMME_dataset = ray.data.from_items(videoMME_data).map(VideoMME_parse)

    #ChartQA
    # with open(FLAGS.ChartQA_path, 'r') as file_:
    #         ChartQA_data = json.load(file_)["annotations"]
    # ChartQA_dataset = ray.data.from_items(ChartQA_data).map(ChartQA_parse)

    if not os.path.exists(FLAGS.output_path):
        os.mkdir(FLAGS.output_path)

    model_paths = [val for val in os.listdir(model_folder) if val.startswith("global_step")]
    model_paths = sorted(model_paths, key=lambda x: int(x[11:]))
    print(f"all model checkpoints are {model_paths}")

    for model_path in model_paths:
        cur_step = int(model_path[11:])
        cur_step = 21000
        last_steps = []
        if cur_step not in last_steps:
            last_steps.append(cur_step)
            step_folder = os.path.join(model_folder, model_path)
            print(f"evaluate dataset for {model_path} in {model_folder}")
            # transform model to vllm format
            checkpoint_model = torch.load(os.path.join(step_folder, "mp_rank_00_model_states.pt"), map_location="cpu")
            torch.save(checkpoint_model["module"], os.path.join("vllm_model_ray", "pytorch_model.bin"))
            #Input the model name or path. Can be GPTQ or AWQ models.
            # MMMU
            # MMMU_dataset_response = MMMU_dataset.map_batches(
            #                             LLMPredictor,
            #                             fn_constructor_kwargs=fn_constructor_kwargs,
            #                             # Set the concurrency to the number of LLM instances.
            #                             concurrency=4,
            #                             batch_size=32,
            #                             # Specify the batch size for inference.
            #                             **resources_kwarg,
            #                         ).take_all()
            # rsp, anw = infer_and_eval(MMMU_dataset_response, FLAGS.output_path, model_path, is_random=False, dataset_name="MMMU") 
            # eval_data = MainEvalOnly(rsp)
            # result = eval_data.eval()
            # print(f"MMMU dataset eval result for {model_path} in {model_folder}: {result}")
            # writer.add_scalar(f'{model_name}_MMMU_val_acc', result["acc"], cur_step)

            # VideoMME
            VideoMME_dataset_response = VideoMME_dataset.map_batches(
                                        LLMPredictor,
                                        fn_constructor_kwargs=fn_constructor_kwargs,
                                        # Set the concurrency to the number of LLM instances.
                                        concurrency=20,
                                        batch_size=400,
                                        # Specify the batch size for inference.
                                        **resources_kwarg,
                                    ).take_all()
            rsp, anw = infer_and_eval(VideoMME_dataset_response, FLAGS.output_path, model_path, is_random=False, dataset_name="VideoMME")
            acc = get_acc(anw, rsp)
            print(f"VideoMME dataset eval result for {model_path} in {model_folder}: {acc}")
            writer.add_scalar(f'{model_name}_VideoMME_val_acc', acc, cur_step)

            # ChartQA
            # ChartQA_dataset_response = ChartQA_dataset.map_batches(
            #                             LLMPredictor,
            #                             fn_constructor_kwargs=fn_constructor_kwargs,
            #                             # Set the concurrency to the number of LLM instances.
            #                             concurrency=2,
            #                             batch_size=40,
            #                             # Specify the batch size for inference.
            #                             **resources_kwarg,
            #                         ).take_all()
            # rsp, anw = infer_and_eval(ChartQA_dataset_response, FLAGS.output_path, model_path, is_random=False, dataset_name="ChartQA")
            # acc = get_open_end_res(anw, rsp)
            # print(f"ChartQA dataset eval result for {model_path} in {model_folder}: {acc}")
            # writer.add_scalar(f'{model_name}_VideoMME_val_acc', acc, cur_step)

            #MMBench en
            # text2index = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
            # rsp, anw = infer_and_eval(llm, sampling_params, MMBench_dataset, FLAGS.output_path, model_path, FLAGS.batch_size, text2index=text2index, dataset_name="MMBenchEN")
            # eval_data = MMBenchEvaluation(rsp, FLAGS.benchmark_original_data)
            # result = eval_data.eval()
            # print(f"MMBenchEN dataset eval result for {model_path} in {model_folder}: {result}")
            # writer.add_scalar(f'{model_name}_MMBenchEN_dev_acc', result[-1]/100)
            fw = open(FLAGS.infer_chekpoint_file, "w")
            for step in last_steps:
                fw.write(str(step) + "\n")
            fw.close()
            break

    writer.close()
if __name__ == "__main__":
    app.run(main)
