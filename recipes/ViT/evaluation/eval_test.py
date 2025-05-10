import os
import os.path as osp
import torch
import time
import json
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
from recipes.ViT.training.models import KimiViT, KimiViTSigLIP
from recipes.ViT.data.dataset import build_dataloader
from recipes.ViT.training.lr_scheduler import build_scheduler
from recipes.ViT.training.optimizer import build_optimizer
from recipes.ViT.helpers.monitor import build_monitor
from deepspeed.ops.adam import FusedAdam
from recipes.ViT.training.models.siglip.processing_siglip import SiglipProcessor
logger = logging.getLogger(__name__)
import pdb
from datasets import load_dataset
from tqdm.auto import tqdm
from torch.utils.data.distributed import DistributedSampler
from recipes.ViT.helpers.hook.utils import process_vision_info


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
    
    def collect(self, outputs, rets, elapsed, **kwargs):
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
            **kwargs
        )


def check_config(args, config):
    config.output_dir = args.output_dir
    config.model.packing = config.dataset.packing

    if config.dataset.num_workers != config.dataset.loader.num_workers:
        config.dataset.num_workers = config.dataset.loader.num_workers
        logger.warning(f"Divergence of 'config.dataset.num_workers' and 'config.dataset.loader.num_workers', rewrite 'config.dataset.num_workers' to {config.dataset.loader.num_workers}")

    model_config_path = osp.join(config.model.dir, "config.json")
    model_config = json.load(open(model_config_path, "r", encoding="utf-8"))
    patch_size = model_config["vision_config"]["patch_size"]
    config.dataset.packing.patch_size = patch_size
    logger.warning(f"Set patch_size = {patch_size} from model config file {model_config_path}")



def load_deepspeed_checkpoint(checkpoint_dir, model, tag="global_step1000"):
    client_state = model.load_checkpoint(
        load_dir=checkpoint_dir,
        tag=tag,  # 自动查找最新检查点
        load_module_only=True,        # 仅加载模型参数
        load_optimizer_states=False,  # 不加载优化器状态
        load_lr_scheduler_states=False
    )
    return model.module  # 返回基础模型对象


def resize(image):
    vision_infos = [
        {
            "type": "image",
            "image": image
        }
    ]
    vision_infos = process_vision_info(vision_infos, 14, 784, 802816)
    return vision_infos[0]


def collate_fn(batch):
    images = batch["image"]
    images = torch.concat(images, dim=0)
    batch["image"] = images.unsqueeze(0)
    batch["label"] = torch.stack(batch["label"], dim=0)
    batch[""] = None

class ImageNetDataset(torch.utils.data.Dataset):

    def __init__(self,
                 data,
                 test_model,
                 image_key,
                 label_key,
                 config):

        grouped_data = []
        
        for idx, row in enumerate(tqdm(data)):
            grouped_data.append((row[image_key], row[label_key]))

        self.config = config
        self.data = grouped_data
        print("data", len(self.data))
        self.processor = None
        self.test_model = test_model

    def __getitem__(self, idx):
        config = self.config
        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(self.test_model, trust_remote_code=True)
            self.processor = SiglipProcessor.from_pretrained(config.model.siglip_dir)

        PIL_image, label = self.data[idx]

        PIL_image = PIL_image.convert("RGB")
        # PIL_image = PIL_image.resize((378, 378))
        PIL_image = resize(PIL_image)
        processed_image = self.processor(images=PIL_image, return_tensors="pt", do_resize=False)["pixel_values"][0]
        from einops import rearrange
        processed_image = rearrange(processed_image, "c (h p1) (w p2) -> (h w) c p1 p2", p1=14, p2=14)
        # processed_image = processed_image.unsqueeze(0)
        assert processed_image.dim() == 4, processed_image.shape
        instance = {}
        instance["image"] = processed_image
        instance["label"] = torch.tensor([label])
        instance["image_position_ids"] = torch.arange(processed_image.shape[0])
        instance["sample_indices"] = torch.zeros((processed_image.shape[0], ), dtype=torch.long)
        width, height = PIL_image.size
        instance["image_grid_thw"] = [(1, height // 14, width // 14)]

        return instance

    def __len__(self):
        return len(self.data)


def create_dataloader(dataset, batch_size=256):
    print("dataset config, num_replicas {}, rank {}".format(dist.get_world_size(), dist.get_rank()))
    sampler = DistributedSampler(dataset,
                                num_replicas=dist.get_world_size(),
                                rank=dist.get_rank(),
                                shuffle=False,
                                seed=42)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )


def train(args):

    deepspeed.init_distributed()

    config = OmegaConf.load(args.config_file)
    print("CJX evaluation", config)
    check_config(args, config)
    
    ctx = DistributedContext(args=args, config=config).setup()
    
    dataset = load_dataset("/llm_reco/caojiangxia/dataset_cache/imagenet-1k", trust_remote_code=True)
    train_ds = dataset["validation"].select(list(range(50000)))

    if args.test_model == "ours":
        test_model = "/llm_reco/liuyang76/Models/siglip2-so400m-patch14-384/"
    else:
        test_model = args.test_model

    train_image_ds = ImageNetDataset(train_ds, test_model, "image", "label", config)

    with deepspeed.zero.Init(config_dict_or_path=args.deepspeed_config, enabled=False):
        
        if args.test_model == "ours":
            model = KimiViTSigLIP(config.model, ctx)
            processor = AutoProcessor.from_pretrained(test_model, trust_remote_code=True)
            print("test model now: ", test_model)
        else:
            model = AutoModel.from_pretrained(test_model, trust_remote_code=True)
            processor = AutoProcessor.from_pretrained(test_model, trust_remote_code=True)
            print("test model now: ", test_model)

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

    for step in range(1000, 20000, 1000):
        tag = f"global_step{step}"
        train_loader = create_dataloader(train_image_ds, batch_size=1)
        if args.test_model == "ours":
            load_deepspeed_checkpoint(args.checkpoint_dir, model, tag=tag)

        model.eval()
        print("before dataset, num_replicas {}, rank {} reached".format(dist.get_world_size(), dist.get_rank()))

        all_label_text_v1 = torch.cat([processor(text="This is a photo of {}".format(label_name), padding="max_length", 
                                            truncation=True, max_length=64,
                                            return_tensors="pt")["input_ids"] for label_name in train_ds.features["label"].names]).to(model.device)
        all_label_text_v2 = torch.cat([processor(text="A {}".format(label_name), padding="max_length", 
                                            truncation=True, max_length=64,
                                            return_tensors="pt")["input_ids"] for label_name in train_ds.features["label"].names]).to(model.device)
        all_label_text_v3 = torch.cat([processor(text="{}".format(label_name), padding="max_length", 
                                            truncation=True, max_length=64,
                                            return_tensors="pt")["input_ids"] for label_name in train_ds.features["label"].names]).to(model.device)

        with torch.no_grad():
            if args.test_model == "ours":
                text_features_v1 = model.module.siglip.get_text_features(input_ids=all_label_text_v1)
                text_features_v2 = model.module.siglip.get_text_features(input_ids=all_label_text_v2)
                text_features_v3 = model.module.siglip.get_text_features(input_ids=all_label_text_v3)
            else:
                text_features_v1 = model.module.get_text_features(input_ids=all_label_text_v1)
                text_features_v2 = model.module.get_text_features(input_ids=all_label_text_v2)
                text_features_v3 = model.module.get_text_features(input_ids=all_label_text_v3)

            text_features_v1 = F.normalize(text_features_v1, dim=-1)
            text_features_v2 = F.normalize(text_features_v2, dim=-1)
            text_features_v3 = F.normalize(text_features_v3, dim=-1)

            text_features = torch.stack([text_features_v1, text_features_v2, text_features_v3], dim=0)  # [3, num_classes, dim]

            print(f"Rank {dist.get_rank()}: len(train_loader) = {len(train_loader)}")
            print(f"Rank {dist.get_rank()}: CUDA device = {torch.cuda.current_device()}")
            torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
            
            device = torch.cuda.current_device()
        
            total_correct_v1 = torch.tensor(0).to(device)
            total_correct_v2 = torch.tensor(0).to(device)
            total_correct_v3 = torch.tensor(0).to(device)
            total_samples = torch.tensor(0).to(device)
            
            for eval_step, batch in enumerate(train_loader):
                image = batch["image"]
                label = batch["label"]
                image, label = image.to(device), label.to(device)
                sample_indices = batch["sample_indices"].to(device).squeeze(0)
                image_position_ids = batch["image_position_ids"].to(device).squeeze(0)
                image_grid_thw = batch["image_grid_thw"][0]
                image_grid_thw = [(image_grid_thw[0].item(), image_grid_thw[1].item(), image_grid_thw[2].item())]
                # sample_indices = None
                # image_position_ids = None

                if args.test_model == "ours":
                    image_features = model.module.siglip.get_image_features(
                        pixel_values=image,
                        sample_indices=sample_indices, 
                        image_position_ids=image_position_ids,
                        image_grid_thw=image_grid_thw,
                        interpolate_pos_encoding=True,
                    )
                else:
                    image_features = model.module.get_image_features(pixel_values=image)

                image_features = F.normalize(image_features, dim=-1)

                # logits = (image_features @ text_features_v1.T).softmax(dim=-1)
                # preds = logits.argmax(dim=1)
                # total_correct_v1 += (preds == label).sum().item()
                # del logits, preds  # 手动释放
                # torch.cuda.empty_cache()  # 清空缓存

                # logits = (image_features @ text_features_v2.T).softmax(dim=-1)
                # preds = logits.argmax(dim=1)
                # total_correct_v2 += (preds == label).sum().item()
                # del logits, preds  # 手动释放
                # torch.cuda.empty_cache()  # 清空缓存

                # logits = (image_features @ text_features_v3.T).softmax(dim=-1)
                # preds = logits.argmax(dim=1)
                # total_correct_v3 += (preds == label).sum().item()

                logits = torch.einsum('bd,ncd->bnc', image_features, text_features)  # [batch, 3, num_classes]
                preds = logits.softmax(dim=-1).argmax(dim=-1)[:, :, None]  # [batch, 3]

                # print((preds[:, 0] == label).shape, "ZDJ")
                total_correct_v1 += (preds[:, 0] == label).sum().item()
                total_correct_v2 += (preds[:, 1] == label).sum().item()
                total_correct_v3 += (preds[:, 2] == label).sum().item()

                total_samples += torch.tensor(label.size(0), device=device)

            print("total_correct_v1: ", total_correct_v1)
            print("total_correct_v2: ", total_correct_v2)
            print("total_correct_v3: ", total_correct_v3)
            print("total_samples: ", total_samples)
            print(f"Rank {dist.get_rank()}: is_initialized = {dist.is_initialized()}")

            dist.all_reduce(total_correct_v1, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_correct_v2, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_correct_v3, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)

            # torch.distributed.all_reduce(total_correct, op=torch.distributed.ReduceOp.SUM)
            # torch.distributed.all_reduce(total_samples, op=torch.distributed.ReduceOp.SUM)
            accuracy_v1 = total_correct_v1.item() / total_samples.item()
            accuracy_v2 = total_correct_v2.item() / total_samples.item()
            accuracy_v3 = total_correct_v3.item() / total_samples.item()
            if dist.get_rank() == 0:
                print(f"Tag {tag}")
                print(f"Total sample {total_samples}")
                print(f"Total correct_v1 {total_correct_v1}, Total correct_v1 {total_correct_v2}, Total correct_v1 {total_correct_v3}")
                print(f"Zero-shot v1: {accuracy_v1 * 100:.2f}%, Zero-shot v2: {accuracy_v2 * 100:.2f}%, Zero-shot v3: {accuracy_v3 * 100:.2f}%")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument("--local_rank", type=int, help="Reserved for deepspeed framework")
    parser.add_argument("--checkpoint_dir", default="/llm_reco_ssd/zangdunju/output2/RecoVLM/SigLIP/4.0.0.1", type=str, help="default checkpoint dir")
    # parser.add_argument("--test_model", default="/llm_reco_ssd/zhouyang12/models/clip-vit-base-patch32", type=str, help="default model dir")
    # parser.add_argument("--test_model", default="/llm_reco_ssd/zhouyang12/models/clip-vit-large-patch14-336", type=str, help="default model dir")
    # parser.add_argument("--test_model", default="/llm_reco_ssd/zhouyang12/models/EVA-CLIP-8B-448", type=str, help="default model dir")
    # parser.add_argument("--test_model", default="/llm_reco_ssd/zhouyang12/models/metaclip-b32-400m", type=str, help="default model dir")
    # parser.add_argument("--test_model", default="/llm_reco_ssd/zhouyang12/models/SigLIP-So400M-Patch14-384/", type=str, help="default model dir")
    # parser.add_argument("--test_model", default="/llm_reco/liuyang76/Models/siglip2-so400m-patch14-384", type=str, help="default model dir")
    parser.add_argument("--test_model", default="ours", type=str, help="default model dir")
    parser = deepspeed.add_config_arguments(parser)
    ags = parser.parse_args()
    train(ags)


"""
/llm_reco_ssd/zhouyang12/models/clip-vit-base-patch32
/llm_reco_ssd/zhouyang12/models/clip-vit-large-patch14-336
/llm_reco_ssd/zhouyang12/models/EVA-CLIP-8B-448
/llm_reco_ssd/zhouyang12/models/metaclip-b32-400m
/llm_reco_ssd/zhouyang12/models/SigLIP-So400M-Patch14-384/
/llm_reco/liuyang76/Models/siglip2-so400m-patch14-384
"""