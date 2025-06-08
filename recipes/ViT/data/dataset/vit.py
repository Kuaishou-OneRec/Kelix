from .parquet import ParquetDataset
import random
import json
import logging
import torch.distributed as dist
from torch.utils.data import Dataset, IterableDataset


logger = logging.getLogger(__name__)


class ViTParquetDataset(IterableDataset):
    def __init__(self, sources, num_workers, shuffle_seed=1024, num_epochs=1, **kwargs):
        self.rng = random.Random(shuffle_seed)
        self.num_workers = num_workers
        self.num_epochs = num_epochs
        self.dataset, _ = self._build_source_dataset(sources, **kwargs)

    def _build_source_dataset(self, sources, **kwargs):
        data_file_list = []
        if dist.get_rank() == 0:
            if isinstance(sources, str) and sources.endswith(".json"):
                with open(sources, "r") as fp:
                    data_files = json.loads(fp.read())
                    data_files = [fn for fn in data_files if fn.endswith(".parquet")]
            else:
                raise TypeError("sources must be a string and a path to a json file")

            # repeat
            for i in range(self.num_epochs):
                data_files.sort()
                self.rng.shuffle(data_files)
                data_file_list += [(fn, i) for fn in data_files]
            logger.error(
                f"ViTParquetDataset rank{dist.get_rank()}: ori_file_num={len(data_files)} file_num={len(data_file_list)}")

        t = [data_file_list]
        dist.broadcast_object_list(t, src=0)
        data_file_list = t[0]

        logger.error(f"ViTParquetDataset rank{dist.get_rank()}: file_num={len(data_file_list)}")
        if len(data_file_list) == 0:
            raise ValueError(f"no datafile found!")

        dataset = ParquetDataset(data_file_list, self.num_workers, **kwargs)
        return dataset, -1

    def state_dict(self, ):
        return self.dataset.state_dict()

    def load_state_dict(self, state_dict):
        self.dataset.load_state_dict(state_dict)

    def __iter__(self):
        return self.dataset.__iter__()
