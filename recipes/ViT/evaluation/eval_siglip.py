import os
import os.path as osp
import torch
import time
import json
import deepspeed
from PIL import Image
import torch.nn as nn
import pandas as pd
import shutil
import glob
import transformers
from collections import Counter
import torch.distributed as dist
from transformers import AutoProcessor, AutoModel
import torch.nn.functional as F
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, IterableDataset
from recipes.ViT.helpers.context import Context, DistributedContext
import argparse
import logging
from omegaconf import OmegaConf
from recipes.ViT.training.models import KimiViT, KimiViTSigLIP
from recipes.ViT.data.dataset import build_dataloader
from recipes.ViT.training.lr_scheduler import build_scheduler
from recipes.ViT.training.optimizer import build_optimizer
from recipes.ViT.helpers.monitor import build_monitor
from deepspeed.ops.adam import FusedAdam

logger = logging.getLogger(__name__)


def check_config(args, ctx, config):
    config.output_dir = args.output_dir
    config.model.packing = config.dataset.packing

    if config.dataset.num_workers != config.dataset.loader.num_workers:
        config.dataset.num_workers = config.dataset.loader.num_workers
        logger.warning(
            f"Divergence of 'config.dataset.num_workers' and 'config.dataset.loader.num_workers', rewrite 'config.dataset.num_workers' to {config.dataset.loader.num_workers}")

    model_config_path = osp.join(config.model.siglip_dir, "config.json")
    model_config = json.load(open(model_config_path, "r", encoding="utf-8"))
    patch_size = model_config["vision_config"]["patch_size"]
    config.dataset.packing.patch_size = patch_size
    logger.warning(f"Set patch_size = {patch_size} from model config file {model_config_path}")

    config.dataset.cache_dir = osp.join("/code/data/zdj/eval_cache", osp.basename(osp.dirname(args.eval_dir)),
                                        osp.basename(args.eval_dir))


def build_labels(args):
    class_file = args.class_file
    df = pd.read_parquet(class_file)
    class_dict = dict()
    class_names = list()
    class_ids = list()
    for _, row in df.iterrows():
        class_id = row["class_id"]
        # class_name = row["class_name"].split(",")[0].strip()
        class_name = row["class_name"]
        class_dict[class_name] = class_id
        class_names.append(class_name)
        class_ids.append(class_id)
    return class_dict, class_names, class_ids


def to_cuda(inputs, device=None):
    if device is None:
        device = torch.cuda.current_device()

    if isinstance(inputs, (list, tuple)):
        inputs = list(inputs)
        for idx, item in enumerate(inputs):
            inputs[idx] = to_cuda(item, device)
        return inputs
    elif isinstance(inputs, (dict, transformers.tokenization_utils_base.BatchEncoding)):
        for key in inputs:
            inputs[key] = to_cuda(inputs[key], device)
        return inputs
    elif isinstance(inputs, torch.Tensor):
        return inputs.to(device)
    return inputs


def get_label_embedding(model, label_texts):
    processor = model.processor
    text_model = model.siglip
    inputs = processor(text=label_texts, padding="longest", return_tensors="pt")
    attention_mask = (inputs["input_ids"] != processor.tokenizer.pad_token_id).long()
    inputs["attention_mask"] = attention_mask
    inputs = to_cuda(inputs)
    label_embeds = text_model.get_text_features(**inputs)
    label_embeds = label_embeds / label_embeds.norm(p=2, dim=-1, keepdim=True)
    return label_embeds


def get_image_embedding(model, batch):
    processor = model.processor
    inputs = dict()
    for key in batch:
        if key not in ["images", "texts", "source", "task", "image_indices", "height_position_ids", "width_position_ids", "image_grid_hws"]:
            inputs[key] = to_cuda(batch[key])
    vision_model = model.siglip
    vision_embeds = vision_model.get_image_features(**inputs)
    vision_embeds = vision_embeds / vision_embeds.norm(p=2, dim=-1, keepdim=True)
    return vision_embeds


def load_ckpt(args, ctx, model, ckpt_path):
    eval_dir = args.eval_dir
    ckpt = torch.load(osp.join(ckpt_path, "mp_rank_00_model_states.pt"), map_location="cpu")
    model_params = ckpt["module"]
    if ctx.rank == 0:
        os.makedirs(eval_dir, exist_ok=True)
        torch.save(model_params, osp.join(eval_dir, "pytorch_model.bin"))
        for file in glob.glob(osp.join("/llm_reco/liuyang76/Models/siglip2-so400m-patch14-384/", "*")):
            if osp.isfile(file) and "safetensors" not in file:
                target_path = osp.join(eval_dir, osp.basename(file))
                if not osp.exists(target_path):
                    shutil.copy(file, target_path)
    dist.barrier()
    model.load_state_dict(model_params, strict=True)
    return model


def evaluate(args, ctx, config, model):
    class_names = ctx.class_names
    class_ids = ctx.class_ids
    label_texts = ctx.label_texts

    model = load_ckpt(args, ctx, model, ckpt_path=ctx.ckpt_path)
    model = model.cuda()

    if ctx.rank == 0:
        print("Start evaluate", ctx.ckpt_path)

    with torch.no_grad():
        label_embeds = get_label_embedding(model, label_texts)

    dataloader = build_dataloader(config.dataset, model=model)

    with torch.no_grad():
        total_correct = 0
        total = 0
        for step, batch in tqdm(enumerate(dataloader, 1), postfix="In rank {}...".format(ctx.rank)):
            texts = batch["texts"]
            labels = [class_ids.index(osp.splitext(x)[0].split("_")[-1]) for x in texts]
            labels = torch.LongTensor(labels)
            vision_embeds = get_image_embedding(model, batch)
            preds = torch.einsum("bd,nd->bn", vision_embeds, label_embeds).detach().cpu().argmax(dim=1)
            correct = (preds == labels).long().sum().item()
            batch_size = labels.shape[0]
            total_correct += correct
            total += batch_size
    dist.barrier()
    result_metrics = torch.LongTensor([total_correct, total]).cuda()
    dist.all_reduce(result_metrics, op=dist.ReduceOp.SUM)
    result_metrics = result_metrics.cpu().numpy()

    total_correct = result_metrics[0]
    total = result_metrics[1]
    if ctx.rank == 0:
        print("total_correct", total_correct)
        print("total", total)
        print("accuracy", total_correct / total)
        tb_writer = ctx.tb_writer
        step = ctx.step
        if tb_writer is not None:
            tb_writer.add_scalar(
                "accuracy",
                total_correct / total,
                global_step=step,
                new_style=True
            )


def main(args):
    deepspeed.init_distributed()

    config = OmegaConf.load(args.config_file)
    print("ZDJ", config)

    ctx = DistributedContext(args=args, config=config).setup()
    check_config(args, ctx, config)

    class_dict, class_names, class_ids = build_labels(args)
    label_texts = ["This is a image of {}.".format(label) for label in class_names]
    
    with deepspeed.zero.Init(config_dict_or_path=args.deepspeed_config, enabled=False):
        model = KimiViTSigLIP(config.model, ctx)

    model.eval()

    ctx.update(
        {
            "class_dict": class_dict,
            "class_names": class_names,
            "class_ids": class_ids,
            "label_texts": label_texts,
        }
    )

    if "global_step" in args.ckpt_folder:
        tb_writer = None
        ctx.update(
            {
                "tb_writer": tb_writer,
                "ckpt_path": args.ckpt_folder,
                "step": 0
            }
        )
        evaluate(args, ctx, config, model)
    else:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=osp.join(args.eval_dir, "log"))
        folders = [_folder for _folder in glob.glob(osp.join(args.ckpt_folder, "*")) if osp.isdir(_folder) and "global_step" in _folder]
        folders.sort(key=lambda x: int(osp.basename(x).split("global_step")[-1]))
        if ctx.rank == 0:
            print(folders)
        for folder in folders:
            step = int(osp.basename(folder).split("global_step")[-1])
            ctx.update(
                {
                    "tb_writer": tb_writer,
                    "ckpt_path": folder,
                    "step": step,
                }
            )
            evaluate(args, ctx, config, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str)
    parser.add_argument('--output_dir', type=str, default="")
    parser.add_argument('--eval_dir', type=str)
    parser.add_argument("--ckpt_folder", type=str)
    parser.add_argument("--class_file", type=str, default="/llm_reco/luoxinchen/dataset/ImageNet-1K/imagenet-1k/classes.parquet")
    parser.add_argument("--local_rank", type=int, help="Reserved for deepspeed framework")
    parser = deepspeed.add_config_arguments(parser)
    ags = parser.parse_args()
    main(ags)
