import os
import os.path as osp
import torch
import time
import deepspeed
from PIL import Image
import torch.nn as nn
import torch.distributed as dist
from transformers import AutoProcessor, AutoModel
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, IterableDataset
from recipes.ViT.helpers.context import Context, DistributedContext
import argparse
import logging
from omegaconf import OmegaConf
from recipes.ViT.training.models import MoonViT
from recipes.ViT.data.dataset import build_dataloader
from recipes.ViT.training.lr_scheduler import build_scheduler
from recipes.ViT.training.optimizer import build_optimizer
from recipes.ViT.helpers.monitor import build_monitor
from deepspeed.ops.adam import FusedAdam
logger = logging.getLogger(__name__)


class MonitorDecorator(object):

    def __init__(self, monitor, ctx):
        self.monitor = monitor
        self.model = monitor.model
        self.ctx = ctx
        self.strategy = self.monitor.strategy
        self.inf = 0x3f3f3f3f
    
    def _get_default_init_buffer(self):
        return {
            "step": 0,
            "elapsed": 0.0,
            "world_size": self.ctx.world_size,
            "total_num_samples": 0,
            "total_num_tokens": 0,
            "total_num_valid_tokens": 0,
            "total_text_num_tokens": 0,
            "total_text_num_valid_tokens": 0,
            "total_image_num_tokens": 0,
        }

    @staticmethod
    def calcul_sec_per_step(metric, other):
        metric.buffer["step"] += getattr(other, "step")
        metric.buffer["elapsed"] += getattr(other, "elapsed")
        if metric.buffer["step"] == 0:
            metric.value = 0.
        else:
            metric.value = metric.buffer["elapsed"] / metric.buffer["step"]

    @staticmethod
    def calcul_tokens_per_sec_per_gpu(metric, other):
        metric.buffer["total_num_tokens"] += getattr(other, "total_num_tokens")
        metric.buffer["elapsed"] += getattr(other, "elapsed")
        metric.buffer["world_size"] = getattr(other, "world_size")
        if metric.buffer["elapsed"] == 0 or metric.buffer["world_size"] == 0:
            metric.value = 0.
        else:
            metric.value = metric.buffer["total_num_tokens"] / metric.buffer["elapsed"] / metric.buffer["world_size"]

    @staticmethod
    def calcul_samples_per_sec_per_gpu(metric, other):
        metric.buffer["total_num_samples"] += getattr(other, "total_num_samples")
        metric.buffer["elapsed"] += getattr(other, "elapsed")
        metric.buffer["world_size"] = getattr(other, "world_size")
        if metric.buffer["elapsed"] == 0 or metric.buffer["world_size"] == 0:
            metric.value = 0.
        else:
            metric.value = metric.buffer["total_num_samples"] / metric.buffer["elapsed"] / metric.buffer["world_size"]

    @staticmethod
    def calcul_valid_tokens_per_sec_per_gpu(metric, other):
        pass

    @staticmethod
    def calcul_valid_text_token_ratio(metric, other):
        metric.buffer["total_text_num_tokens"] += getattr(other, "total_text_num_tokens")
        metric.buffer["total_text_num_valid_tokens"] += getattr(other, "total_text_num_valid_tokens")
        if metric.buffer["total_text_num_tokens"] == 0:
            metric.value = 1.
        else:
            metric.value = metric.buffer["total_text_num_valid_tokens"] / metric.buffer["total_text_num_tokens"]

    @staticmethod
    def calcul_valid_token_ratio(metric, other):
        return
        metric.buffer["total_num_tokens"] += getattr(other, "total_num_tokens")
        metric.buffer["total_num_valid_tokens"] += getattr(other, "total_num_valid_tokens")
        if metric.buffer["total_num_tokens"] == 0:
            metric.value = 1.
        else:
            metric.value = metric.buffer["total_num_valid_tokens"] / metric.buffer["total_num_tokens"]

    def register_metrics(self, config):
        monitor = self.monitor
        monitor.register_metric(
            name="step",
            method="add",
            init_value=0,
            verbose_name="Step",
            report_per_step=self.inf,
            verbose_per_step=config.verbose.verbose_per_step
        )
        for name in ["loss", "learning_rate", "grad_norm"]:
            monitor.register_metric(
                name=name,
                method="assign",
                report_name="training/{}".format(name),
                report_per_step=config.report.report_per_step,
                verbose_per_step=config.verbose.verbose_per_step
            )

        for name in ["sec_per_step", "tokens_per_sec_per_gpu", "samples_per_sec_per_gpu"]:
            monitor.register_metric(
                name=name,
                method=getattr(self, "calcul_{}".format(name)),
                init_buffer=self._get_default_init_buffer(),
                report_name="pref/{}".format(name),
                report_per_step=config.report.report_per_step,
                verbose_per_step=config.verbose.verbose_per_step,
                reset_step=config.report.report_per_step,
            )

        for name in ["total_image_num_tokens", "total_text_num_tokens", "total_num_tokens", "total_num_samples", "total_text_num_valid_tokens"]:
            monitor.register_metric(
                name=name,
                init_value=0,
                method="add",
                report_name="perf/{}".format(name),
                report_per_step=config.report.report_per_step,
                verbose_per_step=config.verbose.verbose_per_step
            )

        for name in ["valid_text_token_ratio", "valid_token_ratio", "valid_tokens_per_sec_per_gpu"]:
            monitor.register_metric(
                name=name,
                method=getattr(self, "calcul_{}".format(name)),
                init_buffer=self._get_default_init_buffer(),
                report_name="pref/{}".format(name),
                report_per_step=config.report.report_per_step,
                verbose_per_step=config.verbose.verbose_per_step,
                reset_step=config.report.report_per_step,
            )
    
    def collect(self,  rets, elapsed, **kwargs):
        model = self.model
        monitor = self.monitor
        ctx = self.ctx
        loss = rets.loss
        total_image_num_tokens = rets.total_image_num_tokens
        total_text_num_tokens = rets.total_text_num_tokens
        total_num_tokens = total_image_num_tokens + total_text_num_tokens
        total_text_num_valid_tokens = rets.total_text_num_valid_tokens
        total_num_samples = rets.total_num_samples

        token_metrics = torch.tensor([total_image_num_tokens, total_text_num_tokens, total_num_tokens, total_num_samples, total_text_num_valid_tokens]).cuda()
        dist.all_reduce(token_metrics, op=dist.ReduceOp.SUM)

        total_image_num_tokens = token_metrics[0].cpu().item()
        total_text_num_tokens = token_metrics[1].cpu().item()
        total_num_tokens = token_metrics[2].cpu().item()
        total_num_samples = token_metrics[3].cpu().item()
        total_text_num_valid_tokens = token_metrics[4].cpu().item()

        return Context(
            step=1,
            loss=loss.detach().cpu().item(),
            learning_rate=model.lr_scheduler.get_lr()[0],
            grad_norm=model.get_global_grad_norm().detach().cpu().item(),
            elapsed=elapsed,
            world_size=ctx.world_size,
            total_image_num_tokens=total_image_num_tokens,
            total_text_num_tokens=total_text_num_tokens,
            total_text_num_valid_tokens=total_text_num_valid_tokens,
            total_num_samples=total_num_samples,
            total_num_tokens=total_num_tokens,
        )


def check_config(args, config):
    config.output_dir = args.output_dir
    if config.dataset.num_workers != config.dataset.loader.num_workers:
        config.dataset.num_workers = config.dataset.loader.num_workers
        logger.warning(f"Divergence of 'config.dataset.num_workers' and 'config.dataset.loader.num_workers', rewrite 'config.dataset.num_workers' to {config.dataset.loader.num_workers}")


def train(args):


    deepspeed.init_distributed()

    config = OmegaConf.load(args.config_file)
    print("ZDJ", config)
    check_config(args, config)
    
    ctx = DistributedContext(args=args, config=config).setup()
    
    with deepspeed.zero.Init(config_dict_or_path=args.deepspeed_config, enabled=False):
        model = KimiViViT(config.model, ctx)
    optimizer = build_optimizer(config.optimizer, model, model_name="siglip")
    optimizer = FusedAdam(model.parameters(),
                        lr=config.optimizer.learn_rate,
                        betas=(0.9, 0.95),
                        eps=1.0e-8)
    lr_scheduler = build_scheduler(config.lr_scheduler, optimizer)

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        args=args,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler
    )

    model.train()

    dataloader = build_dataloader(config.dataset)

    monitor = build_monitor(config, ctx, model=model, dataloader=dataloader)
    decorator = MonitorDecorator(monitor, ctx)
    decorator.register_metrics(config)

    start = time.time()
    for step, batch in enumerate(dataloader, 1):

        images = batch["images"]
        texts = batch["texts"]
        pooler, text_embeds, rets = model(images=images, texts=texts)
        
        loss = rets.loss

        model.backward(loss)

        model.step()
        end = time.time()
        package = decorator.collect(rets, end - start)
        monitor.step(package)
        start = end


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument("--local_rank", type=int, help="Reserved for deepspeed framework")
    parser = deepspeed.add_config_arguments(parser)
    ags = parser.parse_args()
    train(ags)
