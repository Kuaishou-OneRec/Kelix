import os
import os.path as osp
import torch
import time
import json
import deepspeed
from PIL import Image
import torch.nn as nn
from collections import Counter
import torch.distributed as dist
from transformers import AutoProcessor, AutoModel
import torch.nn.functional as F
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
        for name in ["loss", "learning_rate", "grad_norm", 'AR_loss', 'Contrastive_loss']:
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
                report_name="perf/{}".format(name),
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
                report_name="perf/{}".format(name),
                report_per_step=config.report.report_per_step,
                verbose_per_step=config.verbose.verbose_per_step,
                reset_step=config.report.report_per_step,
            )

    def montior_dataset(self, rets):

        monitor = self.monitor
        if "source" in rets:
            count_format = "source/{}"
            sources = [count_format.format(src) for src in rets.source]
            sources_list = [None for _ in range(self.ctx.world_size)]
            torch.all_gather_object(sources_list, sources)
            tmp_sources = list()
            for iter_sources in sources_list:
                tmp_sources.extend(iter_sources)
            sources = tmp_sources

            source_dict = dict(Counter(sources))
            for source_name in source_dict:
                dataset_name = source_name
                if dataset_name not in monitor.metrics_names:
                    monitor.register_metric(
                        name=dataset_name,
                        init_value=0,
                        method="add",
                        report_per_step=config.report.report_per_step,
                        verbose_per_step=self.inf,
                        can_skip_update=True
                    )
            return source_dict

        return dict()

    def collect(self, outputs, rets, elapsed, **kwargs):
        model = self.model
        monitor = self.monitor
        ctx = self.ctx
        loss = rets.loss
        AR_loss = rets.AR_loss
        Contrastive_loss = rets.Contrastive_loss

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
            AR_loss=AR_loss.detach().cpu().item(),
            Contrastive_loss=Contrastive_loss.detach().cpu().item(),
            learning_rate=model.lr_scheduler.get_lr()[0],
            grad_norm=model.get_global_grad_norm().detach().cpu().item(),
            elapsed=elapsed,
            world_size=ctx.world_size,
            total_image_num_tokens=total_image_num_tokens,
            total_text_num_tokens=total_text_num_tokens,
            total_text_num_valid_tokens=total_text_num_valid_tokens,
            total_num_samples=total_num_samples,
            total_num_tokens=total_num_tokens,
            **kwargs,
            **self.montior_dataset(rets),
        )


def check_config(args, config):
    config.output_dir = args.output_dir
    config.model.packing = config.dataset.packing

    if config.dataset.num_workers != config.dataset.loader.num_workers:
        config.dataset.num_workers = config.dataset.loader.num_workers
        logger.warning(f"Divergence of 'config.dataset.num_workers' and 'config.dataset.loader.num_workers', rewrite 'config.dataset.num_workers' to {config.dataset.loader.num_workers}")

    model_config_path = osp.join(config.model.siglip_dir, "config.json")
    model_config = json.load(open(model_config_path, "r", encoding="utf-8"))
    patch_size = model_config["vision_config"]["patch_size"]
    config.dataset.packing.patch_size = patch_size
    logger.warning(f"Set patch_size = {patch_size} from model config file {model_config_path}")


def train(args):

    deepspeed.init_distributed()

    config = OmegaConf.load(args.config_file)
    print("ZDJ", config)
    check_config(args, config)
    
    ctx = DistributedContext(args=args, config=config).setup()
    
    with deepspeed.zero.Init(config_dict_or_path=args.deepspeed_config, enabled=False):
        model = KimiViTSigLIP(config.model, ctx)
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

    # freeze LLN
    model.text_decoder.requires_grad_(False)

    model.train()

    dataloader = build_dataloader(config.dataset, model=model)

    monitor = build_monitor(config, ctx, model=model, dataloader=dataloader)
    decorator = MonitorDecorator(monitor, ctx)
    decorator.register_metrics(config)

    start = time.time()
    for step, batch in enumerate(dataloader, 1):
        
        images = batch["images"]
        texts = batch["texts"]
        outputs, rets = model(package=batch, images=images, texts=texts)
        
        loss = rets.loss

        model.backward(loss)

        model.step()
        end = time.time()
        package = decorator.collect(outputs, rets, end - start)
        monitor.step(package)
        start = end
    
    monitor.step(force_save=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument("--local_rank", type=int, help="Reserved for deepspeed framework")
    parser = deepspeed.add_config_arguments(parser)
    ags = parser.parse_args()
    train(ags)
