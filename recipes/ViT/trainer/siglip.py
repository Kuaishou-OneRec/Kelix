import os
import os.path as osp
import torch
import deepspeed
from PIL import Image
import torch.nn as nn
import torch.distributed as dist
from transformers import AutoProcessor, AutoModel
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, IterableDataset
from recipes.ViT.context import Context, DistributedContext
import argparse
from omegaconf import OmegaConf
from recipes.ViT.models import KimiViT
from recipes.ViT.dataset import build_dataloader
from recipes.ViT.lr_scheduler import build_scheduler
from recipes.ViT.optimizer import build_optimizer
from recipes.ViT.monitor import build_monitor


def check_config(config):
    pass


def register_metrics(config, monitor):
    inf = 0x3f3f3f3f
    monitor.register_permanent_metric(
        name="Step",
        init_value=0,
        report_per_step=inf,
        verbose_per_step=config.verbose.verbose_per_step
    )
    for name in ["loss", "learning_rate", "grad_norm"]:
        monitor.register_interim_metrics(
            name=name,
            report_name="training/{}".format(name),
            report_per_step=config.report.report_per_step,
            verbose_per_step=config.verbose.verbose_per_step
        )

    for name in ["sec_per_step", "tokens_per_sec_per_gpu", "samples_per_sec_per_gpu"]:
        monitor.register_interim_metrics(
            name=name,
            report_name="pref/{}".format(name),
            report_per_step=config.report.report_per_step,
            verbose_per_step=config.verbose.verbose_per_step
        )

    for name in ["total_num_tokens", "total_num_samples", "total_num_valid_tokens"]:
        monitor.register_permanent_metric(
            name=name,
            init_value=0,
            report_name="perf/{}".format(name),
            report_per_step=config.report.report_per_step,
            verbose_per_step=config.verbose.verbose_per_step
        )

    for name in ["valid_tokens_per_sec_per_gpu", "valid_token_ratio"]:
        monitor.register_interim_metrics(
            name=name,
            report_name="pref/{}".format(name),
            report_per_step=config.report.report_per_step,
            verbose_per_step=config.verbose.verbose_per_step
        )


def train(args):
    config = OmegaConf.load(args.config_file)

    deepspeed.init_distributed()

    check_config(config)
    ctx = DistributedContext(args=args, config=config).setup()

    with deepspeed.zero.Init(config_dict_or_path=args.deepspeed_config, enabled=False):
        model = KimiViT(config.model, ctx)
    monitor = build_monitor(config, ctx)
    register_metrics(config, monitor)

    optimizer = build_optimizer(config.optimizer, model, model_name="siglip")
    lr_scheduler = build_scheduler(config.lr_scheduler, optimizer)

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        args=args,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler
    )

    dataloader = build_dataloader(config.dataset)

    model.train()

    for step, batch in enumerate(dataloader, 1):
        images = batch["images"]
        texts = batch["texts"]
        outputs, loss = model(images=images, texts=texts)
        
        model.backward(loss)
        out = Context(
            Step=1,
            loss=loss.detach().cpu().item(),
            learning_rate=1.0,
            grad_norm=1,
            sec_per_step=1,
            tokens_per_sec_per_gpu=2,
            samples_per_sec_per_gpu=3,
            total_num_tokens=3,
            total_num_samples=4,
            total_num_valid_tokens=5,
            valid_tokens_per_sec_per_gpu=10,
            valid_token_ratio=11,
        )
        monitor.step(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument("--local_rank", type=int, help="Reserved for deepspeed framework")
    parser = deepspeed.add_config_arguments(parser)
    ags = parser.parse_args()
    train(ags)
