import io
import os
import torch
import json
import uuid
import torch
import decord
import base64
import logging
import traceback
import numpy as np
import pandas as pd
from copy import deepcopy
import pyarrow as pa
import torch.nn as nn
import os.path as osp
from PIL import Image
import multiprocessing
from io import BytesIO
import pyarrow.parquet as pq
from torch.utils.data import Dataset, IterableDataset, DataLoader
from recipes.ViT.helpers.hook import build_hook
from typing import Union, Iterable, Optional, List, Dict, Tuple, Any
logger = logging.getLogger(__name__)


class ParquetDataset(IterableDataset):
    def __init__(self, data_files, num_workers, **kwargs):
        self.data_files = data_files
        self.num_workers = num_workers
        model = kwargs.pop("model")
        self.batch_size = kwargs.get("loader")["batch_size"]
        packing_kwargs = kwargs.get("packing")
        self.use_packing = packing_kwargs.enabled
        self.patch_size = packing_kwargs.patch_size
        self.packing_drop_ratio = packing_kwargs.drop_ratio
        self.packing_max_length = packing_kwargs.max_length

        self.after_hook = list()
        self.before_hook = list()

        for hook_info in kwargs.get("hooks"):
            position = hook_info.get("position")
            if position == "after":
                hook_list = self.after_hook
            else:
                hook_list = self.before_hook
            hook = build_hook(processor=model.processor, type=hook_info["type"], **kwargs)

            hook_list.append(hook)

        manager = multiprocessing.Manager()

        self.finish_dict_all = manager.dict()
        self.offset_dict_all = manager.dict()
        for i in range(self.num_workers):
            self.finish_dict_all[i] = manager.dict()
            self.offset_dict_all[i] = manager.dict()

    @staticmethod
    def pytorch_worker_info(group=None):
        rank, world_size, worker, num_workers = 0, 1, 0, 1
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
        else:
            try:
                import torch.distributed as dist

                if dist.is_available() and dist.is_initialized():
                    group = group or dist.group.WORLD
                    rank = dist.get_rank(group=group)
                    world_size = dist.get_world_size(group=group)
            except ModuleNotFoundError:
                pass
        if "WORKER" in os.environ and "NUM_WORKERS" in os.environ:
            worker = int(os.environ["WORKER"])
            num_workers = int(os.environ["NUM_WORKERS"])
        else:
            try:
                import torch.utils.data

                worker_info = torch.utils.data.get_worker_info()
                if worker_info is not None:
                    worker = worker_info.id
                    num_workers = worker_info.num_workers
            except ModuleNotFoundError:
                pass

        return rank, world_size, worker, num_workers

    def state_dict(self, ):
        rank, world_size, worker, num_workers = self.pytorch_worker_info()

        state_dict = {
            "finish_dict": dict(self.finish_dict_all[worker]),
            "offset_dict": dict(self.offset_dict_all[worker])
        }
        return state_dict

    def load_state_dict(self, state_dict):
        rank, world_size, worker, num_workers = self.pytorch_worker_info()
        finish_dict = state_dict["finish_dict"]
        offset_dict = state_dict["offset_dict"]

        # support old ckpt format
        tmp_finish_dict = dict()
        tmp_offset_dict = dict()

        for k, v in finish_dict.items():
            if isinstance(k, str):
                tmp_finish_dict[(k, 0)] = v
            elif isinstance(k, tuple) and len(k) == 2:
                tmp_finish_dict[k] = v
            else:
                raise NotImplementedError(f"Unsupported dataloader checkpoint format.")

        for k, v in offset_dict.items():
            if isinstance(k, str):
                fn, group_idx = k.split("|")
                group_idx = int(group_idx)
                tmp_offset_dict[(fn, 0, group_idx)] = v
            elif isinstance(k, tuple) and len(k) == 3:
                tmp_offset_dict[k] = v
            else:
                raise NotImplementedError(f"Unsupported dataloader checkpoint format.")

                # clear cur state
        self.finish_dict_all[worker].clear()
        self.offset_dict_all[worker].clear()

        # update
        self.finish_dict_all[worker].update(tmp_finish_dict)
        self.offset_dict_all[worker].update(tmp_offset_dict)
        logger.warning(f"[rank{rank}-worker{worker}] load checkpoint success.")

    def _parser(self, sample, url):
        try:
            key = sample.pop("uuid")
            processed_sample = {
                "__key__": key,
                "__url__": url,
                "json": sample
            }
            return processed_sample

        except:
            logger.error(f"ParquetDataset parse sample error!!! err_msg={traceback.format_exc()}")
            return None

    def __iter__(self, ):
        rank, world_size, worker, num_workers = self.pytorch_worker_info()
        assert num_workers == self.num_workers, "{} {}".format(num_workers, self.num_workers)

        finish_dict = self.finish_dict_all[worker]
        offset_dict = self.offset_dict_all[worker]

        total_num_workers = num_workers * world_size
        local_worker_idx = rank * num_workers + worker
        fn_list = self.data_files[local_worker_idx::total_num_workers]
        logger.warning(
            f"ParquetDataset Info: {rank=}, {world_size=}, {worker=}, {num_workers=}, {len(fn_list)=}"
        )

        try:
            for epoch_fn in fn_list:
                fn, epoch_idx = epoch_fn
                if (fn, epoch_idx) in finish_dict:
                    logger.warning(f"[Rank{rank}-{worker}] skip {fn}")
                    continue

                # open parquet file
                try:
                    parquet_file = pq.ParquetFile(fn)
                except Exception as e:
                    logger.error(
                        f"ParquetDataset error, open parquet fail!!! {fn=}, error_msg={traceback.format_exc()}")
                    continue

                # process file content
                logger.warning(f"[Rank{rank}-{worker}] {fn} total row_groups: {parquet_file.num_row_groups}")
                for group_idx in range(parquet_file.num_row_groups):
                    try:
                        offset = 0
                        fn_group_key = (fn, epoch_idx, group_idx)
                        if fn_group_key in offset_dict:
                            if offset_dict[fn_group_key] == -1:
                                logger.warning(f"[Rank{rank}-{worker}] skip {fn}-epoch{epoch_idx}-group{group_idx}")
                                continue
                            else:
                                offset = offset_dict[fn_group_key] + 1

                        row_group = parquet_file.read_row_group(group_idx)
                        if offset >= row_group.num_rows:
                            continue
                        logger.warning(
                            f"[Rank{rank}-{worker}] start {fn}-epoch{epoch_idx}-group{group_idx}-offset{offset}")
                        row_pandas = row_group.to_pandas().reset_index()

                        for row_idx, row in row_pandas.iterrows():
                            if row_idx < offset:
                                continue
                            
                            row_info_str = f"{fn}-epoch{epoch_idx}-group{group_idx}-offset{row_idx}"

                            try:
                                offset_dict[fn_group_key] = row_idx
                                sample = row.to_dict()
                                for hook in self.before_hook:
                                    if sample is not None:
                                        sample = hook(sample, row_info_str)
                                
                                if sample is None:
                                    continue

                                sample = self._parser(sample, fn)

                                if sample is not None:
                                    for hook in self.after_hook:
                                        if sample is not None:
                                            sample = hook(sample, row_info_str)
                                    
                                    if sample is not None:
                                        yield sample
                            except GeneratorExit:
                                # 正确处理生成器退出
                                logger.warning(
                                    f"Generator exited at {fn}-epoch{epoch_idx}-group{group_idx}-row{row_idx}")
                                return
                            except Exception as e:
                                logger.error(f"Error processing row {row_idx}: {str(e)}")
                                continue

                            if row_idx % 1000 == 0 and row_idx > 0:
                                logger.warning(
                                    f"Processing row {row_idx} in {fn}-epoch{epoch_idx}-group{group_idx}")

                        # group finish
                        logger.warning(f"[Rank{rank}-{worker}] {fn}-epoch{epoch_idx}-group{group_idx} finish.")
                        offset_dict[fn_group_key] = -1

                    except GeneratorExit:
                        # 正确处理生成器退出
                        logger.warning(f"Generator exited during group processing")
                        return
                    except Exception as e:
                        logger.error(f"Error processing group {group_idx}: {str(e)}")
                        continue

                # file finish
                logger.warning(f"[Rank{rank}-{worker}] {fn} finish.")
                finish_dict[(fn, epoch_idx)] = True

        except GeneratorExit:
            # 正确处理生成器退出
            logger.warning("Generator exited during file processing")
            return
        except Exception as e:
            logger.error(f"Error in dataset iterator: {str(e)}\n{traceback.format_exc()}")
            raise