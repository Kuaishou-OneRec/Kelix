import argparse

import time
import os
import glob
import logging
import collections
from typing import List

import torch
import deepspeed
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from transformers import AutoTokenizer, Qwen2VLForConditionalGeneration
from safetensors import safe_open

from qwen_vl_utils import process_vision_info

from datasets import LLaVA_CC3M_Dataset
from merge_checkpoints import convert_zero_checkpoint_to_state_dict

from torch.utils.tensorboard import SummaryWriter

def get_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", type=str, default=None,
                        help="The directory of the pretrained model.")

    parser.add_argument("--dataset", type=str, default=None,
                        help="The path of training data.")

    parser.add_argument("--max_length", type=int, default=1024,
                        help="Max tokens per sentence in corpus")
    
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size per GPU")

    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Number of epochs to train")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Reserved for deepspeed framework")
    
    parser.add_argument("--output_dir", type=str, default=None,
                        help="The directory to write the trained model")
    
    parser.add_argument("--save_checkpoint_per_step", type=int, default=None,
                        help="The number of steps to save a checkpoint")

    parser.add_argument("--save_checkpoint_every_epoch", action="store_true",
                        help="Save checkpoint at the end of every epoch")

    parser.add_argument("--logging_per_step", type=int, default=100,
                        help="The number of steps to log training info")

    parser.add_argument("--merge_checkpoint", action="store_true",
                        help="Merge the checkpoint files into a single file")
    
    parser.add_argument("--merge_checkpoint_dtype", type=str, default="fp16",
                        choices=["fp32", "fp16", "bf16"],
                        help="The dtype of the merged checkpoint file")

    parser.add_argument("--merge_checkpoint_output_file", type=str, default="pytorch_model.bin",
                        help="The name of the merged checkpoint file")
    

    parser.add_argument("--seed", type=int, default=123,
                        help="Manual seed for RNG")

    return parser

def load_safetensors(path):
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    return tensors

def load_zero3_state_dict(model, model_dir):

    missing_keys: List[str] = []
    unexpected_keys: List[str] = []
    error_msgs: List[str] = []

    if dist.get_rank() == 0:
        state_dict = collections.OrderedDict()
        patterns = glob.glob(os.path.join(model_dir, "model-*.safetensors"))
        for model_path in patterns:
            #state_dict.update(torch.load(model_path, map_location="cpu"))
            state_dict.update(load_safetensors(model_path))

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        if metadata is not None:
            # mypy isn't aware that "_metadata" exists in state_dict
            state_dict._metadata = metadata  # type: ignore[attr-defined]
    else:
        state_dict = None
    
    dist.barrier()

    def load(module, local_state_dict, prefix=""):
        # because zero3 puts placeholders in model params, this context
        # manager gathers (unpartitions) the params of the current layer, then loads from
        # the state dict and then re-partitions them again
        with deepspeed.zero.GatheredParameters(list(module.parameters(recurse=False)), modifier_rank=0):
            if dist.get_rank() == 0:
                local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
                print(f"Load: {prefix}")
                module._load_from_state_dict(
                    state_dict, prefix, local_metadata, True,
                    missing_keys, unexpected_keys, error_msgs
                )

        for name, child in module._modules.items():
            if child is not None:
                child_prefix = prefix + name + '.'
                if state_dict:
                    child_state_dict = {
                        k: v for k, v in local_state_dict.items() if \
                            k.startswith(child_prefix)}
                else:
                    child_state_dict = None
                load(child, child_state_dict, child_prefix)

    load(model, state_dict, prefix="")

def print_rank_n(*msg, rank=0):
    if dist.get_rank() == rank:
        print(*msg)

def print_rank_0(*msg):
    print_rank_n(*msg, rank=0)

def move_to_cuda(batch):
    for t in batch.values():
        t.cuda(dist.get_rank())

def train():
    arg_parser = get_argument_parser()
    arg_parser = deepspeed.add_config_arguments(arg_parser)
    args = arg_parser.parse_args()

    assert any([args.save_checkpoint_per_step, args.save_checkpoint_every_epoch]), \
        "The checkpoint saving frequency is not set, save_checkpoint_per_step or " \
        "save_checkpoint_every_epoch should be set."

    deepspeed.init_distributed()

    if dist.get_rank() == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "log"))

    with deepspeed.zero.Init(config_dict_or_path=args.deepspeed_config):
        # TODO: add support for other models
        model_config = Qwen2VLForConditionalGeneration.config_class.from_pretrained(args.model_dir)
        model = Qwen2VLForConditionalGeneration(model_config)

    load_zero3_state_dict(model, args.model_dir)
    model.train()
    model.gradient_checkpointing_enable()
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_engine, _, _, _ = deepspeed.initialize(args=args,
                                                 model=model,
                                                 model_parameters=model_parameters)

    dataset = LLaVA_CC3M_Dataset(
        source=args.dataset,
        processor_path=args.model_dir,
    )
    sampler = DistributedSampler(dataset)
    start_time = time.time()
    for epoch in range(args.num_epochs):
        for batch in torch.utils.data.DataLoader(dataset,
                                                 batch_size=args.batch_size,
                                                 sampler=sampler):
            move_to_cuda(batch)
            input_ids = batch["input_ids"]
            loss_mask = batch["loss_mask"]
            attention_mask = batch.get("attention_mask", None)
            pixel_values = batch.get("pixel_values", None)
            pixel_values_videos = batch.get("pixel_values_videos", None)
            image_grid_thw = batch.get("image_grid_thw", None)
            video_grid_thw = batch.get("video_grid_thw", None)

            input_ids = input_ids * (input_ids > 0).to(torch.int64)
            labels = input_ids * loss_mask + -100 * (1 - loss_mask)

            print_rank_0("input_ids", input_ids, labels)
            loss = model_engine(
                input_ids, labels=labels, attention_mask=attention_mask,
                pixel_values=pixel_values, pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw
            ).loss

            model_engine.backward(loss)

            model_engine.step()
            model_engine.zero_grad()
            iteration = model_engine.global_steps
            if not args.save_checkpoint_every_epoch and \
                iteration % args.save_checkpoint_per_step == 0 and \
                    iteration > 0 and model_engine.is_gradient_accumulation_boundary():
                model_engine.save_checkpoint(save_dir=args.output_dir)

            avg_loss = torch.tensor(loss.item()).cuda()
            dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
            avg_loss = avg_loss.item() / dist.get_world_size()
            if iteration % args.logging_per_step == 0 and dist.get_rank() == 0 and \
                    model_engine.is_gradient_accumulation_boundary():
                learning_rate = model_engine.lr_scheduler.get_lr()[0]
                end_time = time.time()
                sec_per_step = (end_time - start_time) / args.logging_per_step
                start_time = end_time
                log_dict = {
                    "loss": avg_loss,
                    "learning_rate": learning_rate,
                    "grad_norm": model_engine.get_global_grad_norm(),
                    "sec_per_step": sec_per_step
                }
                for name, data in log_dict.items():
                    writer.log(name, data, iteration)

                print(
                    f"Step: {iteration}, Loss: {avg_loss}, Learning Rate: {learning_rate}, "
                    f"Grad Norm: {model_engine.get_global_grad_norm()}, Sec per Step: {sec_per_step}"
                )

        print_rank_0(f"Epoch {epoch} finished, save checkpoint...")
        if args.save_checkpoint_every_epoch:
            model_engine.save_checkpoint(save_dir=args.output_dir)

    if not args.save_checkpoint_every_epoch:
        model_engine.save_checkpoint(save_dir=args.output_dir)

    if args.merge_checkpoint and dist.get_rank() == 0:
        convert_zero_checkpoint_to_state_dict(
            args.output_dir,
            output_file=args.merge_checkpoint_output_file,
            dtype=args.merge_checkpoint_dtype
        )

    if dist.get_rank() == 0:
        logging.info("Training finished!")

if __name__ == "__main__":
    train()
