import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset, DataLoader
from dataset import ParquetDataset
from common.signature import filter_function_arguments


# def build_dataloader(**kwargs):

#     dataset_class = eval(kwargs["type"])
#     init_kwargs = filter_function_arguments(dataset_class.__init__, kwargs, new_obj=True)
#     dataset = dataset_class(**init_kwargs)
#     loader_kwargs = filter_function_arguments(
#         DataLoader.__init__,
#         kwargs["loader"],
#         new_obj=True,
#         exclude_keys=["dataset", "shuffle"]
#     )
#     if "collate" in loader_kwargs:
#         loader_kwargs["collate"] = eval(loader_kwargs["collate"])

#     dataloader = DataLoader(dataset, shuffle=False, **loader_kwargs)
#     return dataloader
