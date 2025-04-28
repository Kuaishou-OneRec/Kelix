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
from recipes.ViT.training.models import KimiViT
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


def load_deepspeed_checkpoint(checkpoint_dir, model):
    client_state = model.load_checkpoint(
        load_dir=checkpoint_dir,
        tag="global_step1800",  # 自动查找最新检查点
        load_module_only=True,        # 仅加载模型参数
        load_optimizer_states=False,  # 不加载优化器状态
        load_lr_scheduler_states=False
    )
    return model.module  # 返回基础模型对象



# def load_deepspeed_checkpoint_v2(load_dir, ctx, model, dataloader=None, force_load=False):
#     if not force_load and not osp.exists(load_dir):
#         logging.warning(f"Checkpoint directory {load_dir} does not exist, skip loading")
#         return

#     # 分布式训练同步：确保所有进程都准备好加载
#     if dist.is_initialized():
#         dist.barrier()

#     # --- 加载模型检查点和客户端状态 ---
#     try:
#         client_state = model.load_checkpoint(
#             load_dir=load_dir,
#             tag=None,  # 自动查找最新检查点
#             load_module_strict=True,  # 严格匹配模型结构
#             load_optimizer_states=False,  # 根据需求调整
#             load_lr_scheduler_states=False
#         )
#     except Exception as e:
#         logging.error(f"Failed to load model checkpoint: {str(e)}")
#         raise
    
#     return model.module


class ImageNetDataset(torch.utils.data.Dataset):

    def __init__(self,
                 data,
                 test_model,
                 image_key,
                 label_key):

        grouped_data = []
        
        for idx, row in enumerate(tqdm(data)):
            grouped_data.append((row[image_key], row[label_key]))

        self.data = grouped_data
        print("data", len(self.data))
        self.processor = None
        self.test_model = test_model

    def __getitem__(self, idx):
        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(self.test_model, trust_remote_code=True)

        PIL_image, label = self.data[idx]

        PIL_image = PIL_image.convert("RGB")
        processed_image = self.processor(images=PIL_image, return_tensors="pt")["pixel_values"][0]

        instance = {}
        instance["image"] = processed_image
        instance["label"] = torch.tensor([label])

        return instance

    def __len__(self):
        return len(self.data)


def create_dataloader(dataset, batch_size=256):
    sampler = DistributedSampler(dataset, shuffle=False)
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
    print("ZDJ", config)
    check_config(args, config)
    
    ctx = DistributedContext(args=args, config=config).setup()
    
    with deepspeed.zero.Init(config_dict_or_path=args.deepspeed_config, enabled=False):
        model = KimiViT(config.model, ctx)
    # optimizer = build_optimizer(config.optimizer, model, model_name="siglip")
    # optimizer = FusedAdam(model.parameters(),
    #                     lr=config.optimizer.learn_rate,
    #                     betas=(0.9, 0.95),
    #                     eps=1.0e-8)
    # lr_scheduler = build_scheduler(config.lr_scheduler, optimizer)

    # model, optimizer, _, lr_scheduler = deepspeed.initialize(
    #     args=args,
    #     model=model,
    #     optimizer=optimizer,
    #     lr_scheduler=lr_scheduler
    # )

    load_deepspeed_checkpoint(args.checkpoint_dir, model)
    # model = load_deepspeed_checkpoint_v2(args.checkpoint_dir, ctx, model)
    # print(model)

    model.eval()

    dataset = load_dataset("/llm_reco/caojiangxia/dataset_cache/imagenet-1k", trust_remote_code=True)

    train_ds = dataset["train"].select(list(range(50)))
    val_ds = dataset["validation"].select(list(range(50)))

    test_model = "/llm_reco_ssd/zhouyang12/models/SigLIP-So400M-Patch14-384/"
    processor = AutoProcessor.from_pretrained(test_model, trust_remote_code=True)

    all_label_text = torch.cat([processor(text="This is a photo of {}".format(label_name), padding="max_length", 
                                        truncation=True, max_length=77,
                                        return_tensors="pt")["input_ids"] for label_name in train_ds.features["label"].names]).to(model.device)


    # image example: <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=640x360 at 0x7FDAD917DE40>

    text_features = model.module.model.get_text_features(input_ids=all_label_text)


    train_ds, val_ds = ImageNetDataset(train_ds, test_model, "image", "label"), ImageNetDataset(val_ds, test_model, "image", "label")
    train_loader = create_dataloader(train_ds, batch_size=2)
    val_loader = create_dataloader(val_ds, batch_size=2)

    print(len(train_ds), "  ", len(val_ds))

    with torch.no_grad():
        total_correct = torch.tensor(0).to(model.device)
        total_samples = torch.tensor(0).to(model.device)
        for batch in train_loader:
            image = batch["image"]
            label = batch["label"]
            image, label = image.to(model.device), label.to(model.device)

            image_features = model.module.model.get_image_features(pixel_values=image)
            
            logits = (image_features @ text_features.T).softmax(dim=-1)
            preds = logits.argmax(dim=1)
            total_correct += (preds == label).sum().item()
            total_samples += torch.tensor(label.size(0), device=model.device)


        torch.distributed.all_reduce(total_correct, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(total_samples, op=torch.distributed.ReduceOp.SUM)
        accuracy = total_correct.item() / total_samples.item()
        print(f"Total sample {total_samples}, Total correct {total_correct}, Zero-shot Accuracy: {accuracy * 100:.2f}%")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument("--local_rank", type=int, help="Reserved for deepspeed framework")
    parser.add_argument("--checkpoint_dir", default="/llm_reco_ssd/caojiangxia/output2/RecoVLM/SigLIP/0.0.0.6/", type=str, help="default checkpoint dir")
    parser = deepspeed.add_config_arguments(parser)
    ags = parser.parse_args()
    train(ags)
