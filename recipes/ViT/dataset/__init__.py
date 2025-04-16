import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset, DataLoader
from recipes.ViT.common import filter_function_arguments
from recipes.ViT.collator import build_collator
from .parquet import ParquetDataset
from .vit import ViTParquetDataset


def build_dataloader(config, monitor=None):

    dataset_class = eval(config.type)
    init_kwargs = filter_function_arguments(dataset_class.__init__, config, new_obj=True, exclude_keys=["type"])

    dataset = dataset_class(**init_kwargs)
    loader_kwargs = filter_function_arguments(
        DataLoader.__init__,
        config.loader,
        new_obj=True,
        exclude_keys=["dataset", "shuffle"]
    )
    if "collate_fn" in loader_kwargs:
        collate_fn = loader_kwargs["collate_fn"]
        loader_kwargs["collate_fn"] = build_collator(monitor=monitor, collate_fn=collate_fn)

    dataloader = DataLoader(dataset, shuffle=False, **loader_kwargs)
    return dataloader
