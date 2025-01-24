"""Batch inference for Qwen2-VL"""
from absl import flags, app

import json
import collections

from tqdm import tqdm
from vllm import LLM, SamplingParams
from benchmark_dataset import MMMUDataset,MMBenchENDataset 
from torch.utils.data import DataLoader
import pandas as pd
import sys
sys.path.insert(0, 'eval/MMMU/mmmu')
sys.path.insert(0, 'eval/MMBench')
from eval.MMMU.mmmu.main_eval_only import MainEvalOnly
from eval.MMBench.mmbench_evaluation_tricky import MMBenchEvaluation
from eval.RealWorldQA.real_world_qa_evaluation_tricky import RealWorldQAEvaluation
from torch.utils.tensorboard import SummaryWriter
import re
import os
import random
import torch

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "model_name_or_path", None, "The path or name of model."
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
    "tp", 2, "The tensor_parallel_size"
)

flags.DEFINE_string(
    "input_path", None, "The system prompt to use."
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
def collate_fn(samples):
    batch = collections.defaultdict(list)
    for sample in samples:
        for key, item in sample.items():
            batch[key].append(item)
    return batch

def infer_and_eval(llm, sampling_params, dataset, output_file, batch_size, text2index=None, is_random=False):
    rsp = {}
    selects = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "G", "K", "L", "M"]
    pattern = r'[^a-zA-Z0-9.]*[A-Z][^a-zA-Z0-9.]*'
    for batch in tqdm(DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)):
        outputs = llm.generate(batch["inputs"], sampling_params)
        for idx, output in enumerate(outputs):
            key = batch["id"][idx]
            if is_random:
                option = batch["options"][idx]
                if option == []:
                    rsp[key] = ""
                else:
                    rsp[key] = selects[random.randint(0, len(option)-1)]
            else:
                if text2index is not None:
                    match_list = re.findall(pattern, output.outputs[0].text)
                    if len(match_list) > 0:
                        rsp[key] = match_list[0]
                    else:
                        rsp[key] = output.outputs[0].text
                    #rsp[key] = text2index[output.outputs[0].text]
                else:
                    match_list = re.findall(pattern, output.outputs[0].text)
                    if len(match_list) > 0:
                        rsp[key] = match_list[0]
                    else:
                        rsp[key] = output.outputs[0].text

    with open(output_file, "w") as file:
        json.dump(rsp, file)
    return rsp

def main(_):
    #writer = SummaryWriter(FLAGS.logging_folder)
    sampling_params = SamplingParams(
        temperature=FLAGS.temperature, top_p=FLAGS.top_p,
        repetition_penalty=FLAGS.repetition_penalty, max_tokens=FLAGS.max_tokens
        )
    # transform model to vllm format
    checkpoint_model = torch.load(os.path.join(FLAGS.model_name_or_path, "mp_rank_00_model_states.pt"), map_location="cpu")
    torch.save(checkpoint_model["module"], os.path.join("vllm_model", "pytorch_model.bin"))
    #Input the model name or path. Can be GPTQ or AWQ models.
    llm = LLM(
        model="vllm_model",
        tensor_parallel_size=FLAGS.tp,
        limit_mm_per_prompt={
            "image": FLAGS.limit_mm_per_prompt,
            "video": FLAGS.limit_mm_per_prompt
        }
        )
    #MMMU
    dataset = MMMUDataset(
    model_name_or_path="vllm_model",
    data_path=FLAGS.input_path
    )
    rsp = infer_and_eval(llm, sampling_params, dataset, os.path.join(FLAGS.output_path, "MMMU_infer.json"), FLAGS.batch_size, is_random=False) 
    eval_data = MainEvalOnly(rsp)
    result = eval_data.eval()
    print(f"MMMU dataset eval result: {result}")
   # writer.add_scalar('MMMU_val_acc', result["acc"])

    #MMBench en
    #dataset = MMBenchENDataset(
    #model_name_or_path=FLAGS.model_name_or_path,
    #data_path=FLAGS.input_path
   # )
   # text2index = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
   # rsp = infer_and_eval(llm, sampling_params, dataset, os.path.join(FLAGS.output_path, "MMBenchEN_infer.json"), FLAGS.batch_size, text2index=text2index)
   # print("rsp len: ", len(rsp))
   # eval_data = MMBenchEvaluation(rsp, FLAGS.benchmark_original_data)
   # result = eval_data.eval()
   # print(result)
   # writer.add_scalar('MMBenchEN_dev_acc', result[-1])

   # writer.close()
if __name__ == "__main__":
    app.run(main)
