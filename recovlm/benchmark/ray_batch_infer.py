"""Batch inference for Qwen2-VL"""
from absl import flags, app

import json
import collections

from tqdm import tqdm
from vllm import LLM, SamplingParams
from ray_benchmark_dataset import * 
from torch.utils.data import DataLoader
import pandas as pd
import sys
sys.path.insert(0, 'eval/MMMU/mmmu')
sys.path.insert(0, 'eval/MMBench')
from eval.MMMU.mmmu.main_eval_only import MainEvalOnly
from eval.MMBench.mmbench_evaluation_tricky import MMBenchEvaluation
from eval.MME_eval import MMEEval
from eval.OCRBench_eval import eval_OCRBench
from eval.cider_eval import Cider
from eval.cider_eval import COCOEvalCap
from eval.benchmark_v21_eval import eval_Benchmark_v21
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
from llm_predict import LLMPredictor
from utils_infer import get_acc, infer_and_eval, dump_predict_answer
from pycocotools.coco import COCO
import time
ray.init(
    runtime_env={
        "env_vars": {"LD_PRELOAD": "/llm_reco_ssd/luoxinchen/libs/libnccl.so.2.21.5.noece.cpu"},
    }
)

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "model_folder", None, "The folder of model."
)

flags.DEFINE_float(
    "top_p", 1.0, "The top_p params"
)

flags.DEFINE_float(
    "temperature", 0, "The temperature params."
)

flags.DEFINE_integer(
    "max_tokens", 512, "The max tokens to generate."
)

flags.DEFINE_integer(
    "tp", 4, "The tensor_parallel_size"
)

flags.DEFINE_integer(
    "GPU_num", 8, "The GPU number"
)

flags.DEFINE_string(
    "MMMU_path", None, "MMMU dataset path"
)

flags.DEFINE_string(
    "VideoMME_path", None, "VideoMME dataset path"
)

flags.DEFINE_string(
    "ChartQA_path", None, "ChartQA dataset path"
)

flags.DEFINE_string(
    "MMBenchEN_path", None, "MMBenchEN dataset path"
)

flags.DEFINE_string(
    "MMBenchCN_path", None, "MMBenchCN dataset path"
)

flags.DEFINE_string(
    "MME_path", None, "MME dataset path"
)

flags.DEFINE_string(
    "MMTBench_path", None, "MMTBench dataset path"
)

flags.DEFINE_string(
    "MMStar_path", None, "MMStar dataset path"
)

flags.DEFINE_string(
    "MathVista_path", None, "MathVista dataset path"
)

flags.DEFINE_string(
    "OCRBench_path", None, "OCRBench dataset path"
)

flags.DEFINE_string(
    "Flickr30k_path", None, "flickr30k dataset path"
)

flags.DEFINE_string(
    "Benchmark_v21_path", None, "Benchmark_v21 dataset path"
)

flags.DEFINE_string(
    "output_path", None, "The path of file to write results." 
)

flags.DEFINE_integer(
    "limit_mm_per_prompt", 10, "The maximum images and videos of mm_input per prompt"
)

flags.DEFINE_integer(
    "batch_size", 40, "The batch size for inference."
)

flags.DEFINE_float(
    "repetition_penalty", 1.05, "The maximum images of mm_input per prompt"
)

flags.DEFINE_string(
    "mmbenchEn_benchmark_original_data", None, "mmbenchEn original benchmark data"
)

flags.DEFINE_string(
    "mmbenchCn_benchmark_original_data", None, "mmbenchCn original benchmark data"
)

flags.DEFINE_string(
    "logging_folder", "log", "logging folder"
)

flags.DEFINE_string(
    "MMMU_infer_chekpoint_file", "", "save infer checkpoint file"
)

flags.DEFINE_string(
    "MMTBench_infer_chekpoint_file", "", "save infer checkpoint file"
)

flags.DEFINE_string(
    "MME_infer_chekpoint_file", "", "save infer checkpoint file"
)

flags.DEFINE_string(
    "MMStar_infer_chekpoint_file", "", "save infer checkpoint file"
)

flags.DEFINE_string(
    "MathVista_infer_chekpoint_file", "", "save infer checkpoint file"
)

flags.DEFINE_string(
    "MMBenchEN_infer_chekpoint_file", "", "save infer checkpoint file"
)

flags.DEFINE_string(
    "MMBenchCN_infer_chekpoint_file", "", "save infer checkpoint file"
)

flags.DEFINE_string(
    "OCRBench_infer_chekpoint_file", "", "save infer checkpoint file"
)

flags.DEFINE_string(
    "Flickr30k_infer_chekpoint_file", "", "save infer checkpoint file"
)

flags.DEFINE_string(
    "Benchmark_v21_infer_chekpoint_file", "", "save infer checkpoint file"
)

flags.DEFINE_integer(
    "infer_MMMU", 0, "infer MMMU dataset"
)

flags.DEFINE_integer(
    "infer_VideoMME", 0, "infer VideoMME dataset"
)

flags.DEFINE_integer(
    "infer_MMTBench", 0, "infer MMMTBench dataset"
)

flags.DEFINE_integer(
    "infer_MME", 0, "infer MME dataset"
)

flags.DEFINE_integer(
    "infer_MMStar", 0, "infer MMStar dataset"
)

flags.DEFINE_integer(
    "infer_MathVista", 0, "infer MathVista dataset"
)

flags.DEFINE_integer(
    "infer_MMBenchEN", 0, "infer MMBenchEN dataset"
)

flags.DEFINE_integer(
    "infer_MMBenchCN", 0, "infer MMBenchCN dataset"
)

flags.DEFINE_integer(
    "infer_OCRBench", 0, "infer OCRBench dataset"
)

flags.DEFINE_integer(
    "infer_Flickr30k", 0, "infer flickr30k dataset"
)

flags.DEFINE_integer(
    "infer_Benchmark_v21", 0, "infer Benchmark_v21 dataset"
)


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
    print(f"all GPUs number is {FLAGS.GPU_num}")
    concurrency_num = int(FLAGS.GPU_num/FLAGS.tp)
    if not os.path.exists(FLAGS.logging_folder):
        os.mkdir(FLAGS.logging_folder)
    writer = SummaryWriter(FLAGS.logging_folder)
    model_folder = FLAGS.model_folder
    last_steps = [] 
    if FLAGS.infer_MMMU == 1 and os.path.exists(FLAGS.MMMU_infer_chekpoint_file):
        fr = open(FLAGS.MMMU_infer_chekpoint_file, "r")
        for line in fr.readlines():
            last_steps.append(int(line.strip()))
    if FLAGS.infer_MME == 1 and os.path.exists(FLAGS.MME_infer_chekpoint_file):
        fr = open(FLAGS.MME_infer_chekpoint_file, "r")
        for line in fr.readlines():
            last_steps.append(int(line.strip()))
    if FLAGS.infer_MMTBench == 1 and os.path.exists(FLAGS.MMTBench_infer_chekpoint_file):
        fr = open(FLAGS.MMTBench_infer_chekpoint_file, "r")
        for line in fr.readlines():
            last_steps.append(int(line.strip()))
    if FLAGS.infer_MMStar == 1 and os.path.exists(FLAGS.MMStar_infer_chekpoint_file):
        fr = open(FLAGS.MMStar_infer_chekpoint_file, "r")
        for line in fr.readlines():
            last_steps.append(int(line.strip()))
    if FLAGS.infer_MathVista == 1 and os.path.exists(FLAGS.MathVista_infer_chekpoint_file):
        fr = open(FLAGS.MathVista_infer_chekpoint_file, "r")
        for line in fr.readlines():
            last_steps.append(int(line.strip()))
    if FLAGS.infer_MMBenchEN == 1 and os.path.exists(FLAGS.MMBenchEN_infer_chekpoint_file):
        fr = open(FLAGS.MMBenchEN_infer_chekpoint_file, "r")
        for line in fr.readlines():
            last_steps.append(int(line.strip()))
    if FLAGS.infer_MMBenchCN == 1 and os.path.exists(FLAGS.MMBenchCN_infer_chekpoint_file):
        fr = open(FLAGS.MMBenchCN_infer_chekpoint_file, "r")
        for line in fr.readlines():
            last_steps.append(int(line.strip()))

    if FLAGS.infer_OCRBench == 1 and os.path.exists(FLAGS.OCRBench_infer_chekpoint_file):
        fr = open(FLAGS.OCRBench_infer_chekpoint_file, "r")
        for line in fr.readlines():
            last_steps.append(int(line.strip()))

    if FLAGS.infer_Flickr30k == 1 and os.path.exists(FLAGS.Flickr30k_infer_chekpoint_file):
        fr = open(FLAGS.Flickr30k_infer_chekpoint_file, "r")
        for line in fr.readlines():
            last_steps.append(int(line.strip()))

    if FLAGS.infer_Benchmark_v21 == 1 and os.path.exists(FLAGS.Benchmark_v21_infer_chekpoint_file):
        fr = open(FLAGS.Benchmark_v21_infer_chekpoint_file, "r")
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

    ################################################################# Get Dataset ##########################################################
    #MMMU
    if FLAGS.infer_MMMU == 1:
        data_files = []
        for subfolder in os.listdir(FLAGS.MMMU_path):
            for subfile in os.listdir(os.path.join(FLAGS.MMMU_path, subfolder)):
                if subfile.startswith("validation"):
                    file_name = os.path.join(FLAGS.MMMU_path, subfolder, subfile)
                    data_files.append("local://" + file_name)
        print(f"data files: {data_files}")
        MMMU_dataset = ray.data.read_parquet(data_files).map(MMMU_parse)

    #VideoMME
    if FLAGS.infer_VideoMME == 1:
        with open(FLAGS.VideoMME_path, 'r') as file_:
                videoMME_data = json.load(file_)["annotations"]
        VideoMME_dataset = ray.data.from_items(videoMME_data).map(VideoMME_parse)

    #ChartQA
    # with open(FLAGS.ChartQA_path, 'r') as file_:
    #         ChartQA_data = json.load(file_)["annotations"]
    # ChartQA_dataset = ray.data.from_items(ChartQA_data).map(ChartQA_parse)

    #TextVQA
    #MME
    if FLAGS.infer_MME == 1:
        with open(FLAGS.MME_path, 'r') as file_:
                MME_data = json.load(file_)["annotations"]
        MME_dataset = ray.data.from_items(MME_data).map(MME_parse)

    #MMT-Bench
    if FLAGS.infer_MMTBench == 1:
        with open(FLAGS.MMTBench_path, 'r') as file_:
                MMTBench_data = json.load(file_)["annotations"]
        MMTBench_dataset = ray.data.from_items(MMTBench_data).map(MMTBench_parse)

    #MMStar
    if FLAGS.infer_MMStar == 1:
        with open(FLAGS.MMStar_path, 'r') as file_:
                MMStar_data = json.load(file_)["annotations"]
        MMStar_dataset = ray.data.from_items(MMStar_data).map(MMStar_parse)

    #MathVista
    if FLAGS.infer_MathVista == 1:
        with open(FLAGS.MathVista_path, 'r') as file_:
                MathVista_data = json.load(file_)["annotations"]
        MathVista_dataset = ray.data.from_items(MathVista_data).map(MathVista_parse)

    #MMBench_EN
    if FLAGS.infer_MMBenchEN == 1:
        MMBenchEN_dataset = ray.data.read_parquet(FLAGS.MMBenchEN_path).map(MMBench_parse)

    #MMBench_CN
    if FLAGS.infer_MMBenchCN == 1:
        MMBenchCN_dataset = ray.data.read_parquet(FLAGS.MMBenchCN_path).map(MMBench_parse)

    #OCRBench
    if FLAGS.infer_OCRBench == 1:
        OCRBench_dataset = ray.data.read_parquet(FLAGS.OCRBench_path).map(OCRBench_parse)

    if FLAGS.infer_Flickr30k == 1:
        with open(FLAGS.Flickr30k_path, 'r') as file_:
            Flickr30k_data = json.load(file_)["annotations"]
        Flickr30k_dataset = ray.data.from_items(Flickr30k_data).map(Flickr30k_parse)

    #Benchmark v21
    if FLAGS.infer_Benchmark_v21 == 1:
        with open(FLAGS.Benchmark_v21_path, 'r') as file_:
                Benchmark_v21_data = json.load(file_)["annotations"]
        Benchmark_v21_dataset = ray.data.from_items(Benchmark_v21_data).map(Benchmark_v21_parse)

############################################################################# Infer ################################################################

    if not os.path.exists(FLAGS.output_path):
        print(f"output path is {FLAGS.output_path}")
        os.mkdir(FLAGS.output_path)

    model_paths = [val for val in os.listdir(model_folder) if val.startswith("global_step")]
    model_paths = sorted(model_paths, key=lambda x: int(x[11:]))
    print(f"all model checkpoints are {model_paths}")

    for model_path in model_paths:
        cur_step = int(model_path[11:])
        if cur_step not in last_steps:
            last_steps.append(cur_step)
            step_folder = os.path.join(model_folder, model_path)
            print(f"evaluate dataset for {model_path} in {model_folder}")
            # transform model to vllm format
            checkpoint_model = torch.load(os.path.join(step_folder, "mp_rank_00_model_states.pt"), map_location="cpu")
            torch.save(checkpoint_model["module"], os.path.join("vllm_model_ray", "pytorch_model.bin"))

            #Input the model name or path. Can be GPTQ or AWQ models.
            # MMMU
            if FLAGS.infer_MMMU == 1:
                MMMU_dataset_response = MMMU_dataset.map_batches(
                                            LLMPredictor,
                                            fn_constructor_kwargs=fn_constructor_kwargs,
                                            # Set the concurrency to the number of LLM instances.
                                            concurrency=concurrency_num,
                                            batch_size=FLAGS.batch_size,
                                            # Specify the batch size for inference.
                                            **resources_kwarg,
                                        ).take_all()
                rsp, anw = infer_and_eval(MMMU_dataset_response, FLAGS.output_path, model_path, is_random=False, dataset_name="MMMU") 
                eval_data = MainEvalOnly(rsp)
                result, correct_keys = eval_data.eval()
                dump_predict_answer(correct_keys, MMMU_dataset_response, FLAGS.output_path, model_path, "MMMU", rsp)
                print(f"MMMU dataset eval result for {model_path} in {model_folder}: {result}")
                writer.add_scalar(f'benchmark/MMMU_val_acc', result["acc"], cur_step)
                fw = open(FLAGS.MMMU_infer_chekpoint_file, "w")
                for step in last_steps:
                    fw.write(str(step) + "\n")
                fw.close()

            # VideoMME
            # VideoMME_dataset_response = VideoMME_dataset.map_batches(
            #                             LLMPredictor,
            #                             fn_constructor_kwargs=fn_constructor_kwargs,
            #                             # Set the concurrency to the number of LLM instances.
            #                             concurrency=concurrency_num,
            #                             batch_size=FLAGS.batch_size,
            #                             # Specify the batch size for inference.
            #                             **resources_kwarg,
            #                         ).take_all()
            # rsp, anw = infer_and_eval(VideoMME_dataset_response, FLAGS.output_path, model_path, is_random=False, dataset_name="VideoMME")
            # acc = get_acc(anw, rsp)
            # print(f"VideoMME dataset eval result for {model_path} in {model_folder}: {acc}")
            # writer.add_scalar(f'{model_name}_VideoMME_val_acc', acc, cur_step)

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
            # acc = get_acc(anw, rsp)
            # print(f"ChartQA dataset eval result for {model_path} in {model_folder}: {acc}")
            # writer.add_scalar(f'{model_name}_ChartQA_val_acc', acc, cur_step)

            # MME
            if FLAGS.infer_MME == 1:
                MME_dataset_response = MME_dataset.map_batches(
                                            LLMPredictor,
                                            fn_constructor_kwargs=fn_constructor_kwargs,
                                            # Set the concurrency to the number of LLM instances.
                                            concurrency=concurrency_num,
                                            batch_size=FLAGS.batch_size,
                                            # Specify the batch size for inference.
                                            **resources_kwarg,
                                        ).take_all()
                rsp, anw = infer_and_eval(MME_dataset_response, FLAGS.output_path, model_path, is_random=False, dataset_name="MME")
                MME_eval_obj = MMEEval()
                score, correct_keys = MME_eval_obj.process_result(rsp, anw)
                dump_predict_answer(correct_keys, MME_dataset_response, FLAGS.output_path, model_path, "MME", rsp)
                print(f"MME dataset eval result for {model_path} in {model_folder}: {score}")
                writer.add_scalar(f'benchmark/MME_val_score', score, cur_step)
                fw = open(FLAGS.MME_infer_chekpoint_file, "w")
                for step in last_steps:
                    fw.write(str(step) + "\n")
                fw.close()

            # # MMTBench
            if FLAGS.infer_MMTBench == 1:
                MMTBench_dataset_response = MMTBench_dataset.map_batches(
                                            LLMPredictor,
                                            fn_constructor_kwargs=fn_constructor_kwargs,
                                            # Set the concurrency to the number of LLM instances.
                                            concurrency=concurrency_num,
                                            batch_size=FLAGS.batch_size,
                                            # Specify the batch size for inference.
                                            **resources_kwarg,
                                        ).take_all()
                rsp, anw = infer_and_eval(MMTBench_dataset_response, FLAGS.output_path, model_path, is_random=False, dataset_name="MMTBench")
                acc, correct_keys = get_acc(anw, rsp)
                dump_predict_answer(correct_keys, MMTBench_dataset_response, FLAGS.output_path, model_path, "MMTBench", rsp)
                print(f"MMTBench dataset eval result for {model_path} in {model_folder}: {acc}")
                writer.add_scalar(f'benchmark/MMTBench_val_acc', acc, cur_step)
                fw = open(FLAGS.MMTBench_infer_chekpoint_file, "w")
                for step in last_steps:
                    fw.write(str(step) + "\n")
                fw.close()

            # # MMStar
            if FLAGS.infer_MMStar == 1:
                MMStar_dataset_response = MMStar_dataset.map_batches(
                                            LLMPredictor,
                                            fn_constructor_kwargs=fn_constructor_kwargs,
                                            # Set the concurrency to the number of LLM instances.
                                            concurrency=concurrency_num,
                                            batch_size=FLAGS.batch_size,
                                            # Specify the batch size for inference.
                                            **resources_kwarg,
                                        ).take_all()
                rsp, anw = infer_and_eval(MMStar_dataset_response, FLAGS.output_path, model_path, is_random=False, dataset_name="MMStar")
                acc, correct_keys = get_acc(anw, rsp)
                dump_predict_answer(correct_keys, MMStar_dataset_response, FLAGS.output_path, model_path, "MMStar", rsp)
                print(f"MMStar dataset eval result for {model_path} in {model_folder}: {acc}")
                writer.add_scalar(f'benchmark/MMStar_val_acc', acc, cur_step)
                fw = open(FLAGS.MMStar_infer_chekpoint_file, "w")
                for step in last_steps:
                    fw.write(str(step) + "\n")
                fw.close()

            # # MathVista
            if FLAGS.infer_MathVista == 1:
                MathVista_dataset_response = MathVista_dataset.map_batches(
                                            LLMPredictor,
                                            fn_constructor_kwargs=fn_constructor_kwargs,
                                            # Set the concurrency to the number of LLM instances.
                                            concurrency=concurrency_num,
                                            batch_size=FLAGS.batch_size,
                                            # Specify the batch size for inference.
                                            **resources_kwarg,
                                        ).take_all()
                rsp, anw = infer_and_eval(MathVista_dataset_response, FLAGS.output_path, model_path, is_random=False, dataset_name="MathVista")
                acc, correct_keys = get_acc(anw, rsp)
                dump_predict_answer(correct_keys, MathVista_dataset_response, FLAGS.output_path, model_path, "MathVista", rsp)
                print(f"MathVista dataset eval result for {model_path} in {model_folder}: {acc}")
                writer.add_scalar(f'benchmark/MathVista_val_acc', acc, cur_step)
                fw = open(FLAGS.MathVista_infer_chekpoint_file, "w")
                for step in last_steps:
                    fw.write(str(step) + "\n")
                fw.close()

            # #MMBench en
            if FLAGS.infer_MMBenchEN == 1:
                MMBenchEN_dataset_response = MMBenchEN_dataset.map_batches(
                                            LLMPredictor,
                                            fn_constructor_kwargs=fn_constructor_kwargs,
                                            # Set the concurrency to the number of LLM instances.
                                            concurrency=concurrency_num,
                                            batch_size=FLAGS.batch_size,
                                            # Specify the batch size for inference.
                                            **resources_kwarg,
                                        ).take_all()
                text2index = {'A': 0, 'B': 1, 'C': 2, 'D': 3 }
                rsp, anw = infer_and_eval(MMBenchEN_dataset_response, FLAGS.output_path, model_path, text2index=text2index, dataset_name="MMBenchEN")
                eval_data = MMBenchEvaluation(rsp, FLAGS.mmbenchEn_benchmark_original_data)
                result, correct_keys = eval_data.eval()
                dump_predict_answer(correct_keys, MMBenchEN_dataset_response, FLAGS.output_path, model_path, "MMBenchEN", rsp)
                print(f"MMBenchEN dataset eval result for {model_path} in {model_folder}: {result}")
                writer.add_scalar(f'benchmark/MMBenchEN_dev_acc', result[-1]/100, cur_step)
                fw = open(FLAGS.MMBenchEN_infer_chekpoint_file, "w")
                for step in last_steps:
                    fw.write(str(step) + "\n")
                fw.close()

            # #MMBench cn
            if FLAGS.infer_MMBenchCN == 1:
                MMBenchCN_dataset_response = MMBenchCN_dataset.map_batches(
                                            LLMPredictor,
                                            fn_constructor_kwargs=fn_constructor_kwargs,
                                            # Set the concurrency to the number of LLM instances.
                                            concurrency=concurrency_num,
                                            batch_size=FLAGS.batch_size,
                                            # Specify the batch size for inference.
                                            **resources_kwarg,
                                        ).take_all()
                text2index = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
                rsp, anw = infer_and_eval(MMBenchCN_dataset_response, FLAGS.output_path, model_path, text2index=text2index, dataset_name="MMBenchCN")
                eval_data = MMBenchEvaluation(rsp, FLAGS.mmbenchCn_benchmark_original_data)
                result, correct_keys = eval_data.eval()
                dump_predict_answer(correct_keys, MMBenchCN_dataset_response, FLAGS.output_path, model_path, "MMBenchCN", rsp)
                print(f"MMBenchCN dataset eval result for {model_path} in {model_folder}: {result}")
                writer.add_scalar(f'benchmark/MMBenchCN_dev_acc', result[-1]/100, cur_step)
                fw = open(FLAGS.MMBenchCN_infer_chekpoint_file, "w")
                for step in last_steps:
                    fw.write(str(step) + "\n")
                fw.close()

            if FLAGS.infer_OCRBench == 1:
                OCRBench_dataset_response = OCRBench_dataset.map_batches(
                                            LLMPredictor,
                                            fn_constructor_kwargs=fn_constructor_kwargs,
                                            # Set the concurrency to the number of LLM instances.
                                            concurrency=concurrency_num,
                                            batch_size=FLAGS.batch_size,
                                            # Specify the batch size for inference.
                                            **resources_kwarg,
                                        ).take_all()
                rsp, anw = infer_and_eval(OCRBench_dataset_response, FLAGS.output_path, model_path, dataset_name="OCRBench")
                data = []
                for i in range(len(rsp.keys())):
                    cur_response = OCRBench_dataset_response[i]
                    cur_row = {}
                    cur_row["question_type"] = cur_response["question_type"]
                    cur_row["dataset"] = cur_response["dataset"]
                    cur_row["answer"] = cur_response["answers"]
                    cur_row["predict"] = rsp[i]
                    cur_row["image"] = cur_response["image"]
                    cur_row["messages"] = cur_response["messages"]
                    data.append(cur_row)
                output_error_data_path = os.path.join(os.path.join(FLAGS.output_path, os.path.join(model_path, "OCRBench")), "predict_error_data.json")
                output_correct_data_path = os.path.join(os.path.join(FLAGS.output_path, os.path.join(model_path, "OCRBench")), "predict_correct_data.json")
                fw_error = open(output_error_data_path, "w")
                fw_correct = open(output_correct_data_path, "w")
                result = eval_OCRBench(data, fw_error, fw_correct)
                fw_error.close()
                fw_correct.close()
                print(f"OCRBench dataset eval result for {model_path} in {model_folder}: {result}")
                writer.add_scalar(f'benchmark/OCRBench_test_score', result, cur_step)
                fw = open(FLAGS.OCRBench_infer_chekpoint_file, "w")
                for step in last_steps:
                    fw.write(str(step) + "\n")
                fw.close()

            # Flickr30k
            if FLAGS.infer_Flickr30k == 1:
                Flickr30k_dataset_response = Flickr30k_dataset.map_batches(
                                            LLMPredictor,
                                            fn_constructor_kwargs=fn_constructor_kwargs,
                                            # Set the concurrency to the number of LLM instances.
                                            concurrency=concurrency_num,
                                            batch_size=FLAGS.batch_size,
                                            # Specify the batch size for inference.
                                            **resources_kwarg,
                                        ).take_all()
                rsp, anw = infer_and_eval(Flickr30k_dataset_response, FLAGS.output_path, model_path, dataset_name="Flickr30k")

                img2messages = {}
                for i in range(len(Flickr30k_dataset_response)):
                    cur_response = Flickr30k_dataset_response[i]
                    img_id = int(cur_response["ids"])
                    if img_id not in img2messages:
                        img2messages[img_id] = {}
                    img2messages[img_id]["answer"] = cur_response["answers"]
                    img2messages[img_id]["messages"] = cur_response["messages"]
                    img2messages[img_id]["predict"] = cur_response["generated_text"] 
                    
                results = []
                for key, val in rsp.items():
                    results.append({
                        'image_id': int(key),
                        'caption': val,
                    })
                results_file = 'tmp.json'
                json.dump(results, open(results_file, 'w'))
                coco = COCO(FLAGS.Flickr30k_path)
                coco_result = coco.loadRes(results_file)
                coco_eval = COCOEvalCap(coco, coco_result)
                coco_eval.evaluate()
                result = coco_eval.eval["CIDEr"] 
                print(f"Flickr30k dataset eval result for {model_path} in {model_folder}: {result}")
                writer.add_scalar(f'benchmark/Flickr30k_test_score', result, cur_step)

                output_data_path = os.path.join(os.path.join(FLAGS.output_path, os.path.join(model_path, "Flickr30k")), "predict_data.json")
                imgToEval = coco_eval.imgToEval
                sorted_imgToEval = sorted(imgToEval.items(), key = lambda kv:(kv[1], kv[0]))
                output_lines = []
                for (key, val) in sorted_imgToEval:
                    cur_row = {}
                    cur_row["image_id"] = key
                    cur_row["answer"] = img2messages[key]["answer"]
                    cur_row["messages"] = img2messages[key]["messages"]
                    cur_row["predict"] = img2messages[key]["predict"]
                    cur_row["cider_score"] = val
                    output_lines.append(cur_row)
                with open(output_data_path, "w") as fw:
                    json.dump(output_lines, fw, indent=4, separators=(',', ':'))

                fw = open(FLAGS.Flickr30k_infer_chekpoint_file, "w")
                for step in last_steps:
                    fw.write(str(step) + "\n")
                fw.close()

            # Benchmark_v21
            if FLAGS.infer_Benchmark_v21 == 1:
                Benchmark_v21_dataset_response = Benchmark_v21_dataset.map_batches(
                                            LLMPredictor,
                                            fn_constructor_kwargs=fn_constructor_kwargs,
                                            # Set the concurrency to the number of LLM instances.
                                            concurrency=concurrency_num,
                                            batch_size=FLAGS.batch_size,
                                            # Specify the batch size for inference.
                                            **resources_kwarg,
                                        ).take_all()
                rsp, anw = infer_and_eval(Benchmark_v21_dataset_response, FLAGS.output_path, model_path, dataset_name="Benchmark_v21")
                output_data_folder = os.path.join(FLAGS.output_path, os.path.join(model_path, "Benchmark_v21"))
                output_data_path = os.path.join(output_data_folder, "infer_for_gpt.json")
                with open(FLAGS.Benchmark_v21_path, 'r') as file_:
                    Benchmark_v21_data = json.load(file_)
                annotations = Benchmark_v21_data["annotations"]
                for i in range(len(annotations)):
                    key = annotations[i]["key"]
                    annotations[i]["model_output"] = rsp[key]
                with open(output_data_path, "w") as fw: 
                    json.dump(Benchmark_v21_data, fw, indent=4, separators=(',', ':'))
                result_dict, correct_keys = eval_Benchmark_v21(output_data_folder, output_data_path)
                dump_predict_answer(correct_keys, Benchmark_v21_dataset_response, FLAGS.output_path, model_path, "Benchmark_v21", rsp)
                print(f"Benchmark_v21 dataset eval result for {model_path} in {model_folder}: {result_dict}")
                for key,val in result_dict.items():
                    writer.add_scalar(f'benchmark/Benchmark_v21/{key}_acc', val, cur_step)
                fw = open(FLAGS.Benchmark_v21_infer_chekpoint_file, "w")
                for step in last_steps:
                    fw.write(str(step) + "\n")
                fw.close()

            break

    writer.close()
if __name__ == "__main__":
    app.run(main)
