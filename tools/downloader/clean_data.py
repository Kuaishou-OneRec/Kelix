import os
import io
import json
import argparse
import torch
from tqdm import tqdm
from datetime import timedelta
import torch.distributed as dist
import webdataset as wds
import wids
from omegaconf import OmegaConf
from filters import create_filter
import traceback

def setup_proc(args):
    rank = int(os.environ.get('OMPI_COMM_WORLD_RANK', "0"))
    world_size = int(os.environ.get('OMPI_COMM_WORLD_SIZE', "1"))
    master_addr, master_port = args.master_addr.split(":")
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    os.environ["LOCAL_RANK"] = os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', "0")
    os.environ["WORLD_SIZE"] = os.environ.get('OMPI_COMM_WORLD_SIZE', "1")
    os.environ["RANK"] = os.environ.get('OMPI_COMM_WORLD_RANK', "0")
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size, timeout=timedelta(seconds=3600*4))

class Worker(object):

    def __init__(self, config):
        self.config = config
        # self.dataset = wds.WebDataset(
        #     config.dataset_urls,
        #     shardshuffle=False,
        #     handler=wds.warn_and_continue,
        #     nodesplitter=wds.split_by_node,
        #     workersplitter=wds.split_by_worker,
        # )
        self.dataset = wids.ShardListDataset(config.dataset_url, transformations=[])
        data_size = len(self.dataset)
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.output_shard_size = config.output_shard_size

        shard_size = data_size // self.world_size
        self.data_range = (self.rank * shard_size, (self.rank + 1) * shard_size)

        self._shard_id = self.rank
        self._cur_size = 0
        self.tarwriter = None
        self.create_new_writer()

        self.filters = [
            create_filter(c.class_name, c.kwargs) for c in config.filters
        ]

        self.reserved_cnt = 0
        self.filtered_cnt = 0
        self.failed_cnt = 0
        if self.rank == 0:
            if not os.path.exists(config.output_dir):
                os.makedirs(config.output_dir)
        dist.barrier()

    def create_new_writer(self):
        if self.tarwriter is not None:
            print(f"{self._cur_fn} is written, failed {self.failed_cnt}, reserved {self.reserved_cnt} filtered {self.filtered_cnt}")
            self.tarwriter.close()
            self._cur_size = 0
            self._shard_id += self.world_size
            self.failed_cnt = 0
            self.filtered_cnt = 0
            self.reserved_cnt = 0
        self._cur_fn = os.path.join(self.config.output_dir, f"{self._shard_id:06d}.tar")
        self.tarwriter = wds.TarWriter(self._cur_fn)

    def write_sample(self, sample):
        self.tarwriter.write(sample)
        self._cur_size += 1
        if self._cur_size >= self.output_shard_size:
            self.create_new_writer()

    def run(self):
        print('datasize', len(self.dataset))
        for i in tqdm(range(*self.data_range), desc=f"RANK[{self.rank}]"):
            try:
                s = self.dataset[i]
                meta = json.load(s['.json'])
                reserve = True
                for f in self.filters:
                    reserve = f(meta) and reserve
                if reserve:
                    sample = {
                        "__key__": f"{self._shard_id:06d}{self._cur_size:04d}",
                        "json": json.dumps(meta),
                        "jpg": s['.jpg'].read(),
                        "txt": meta['caption']
                    }
                    self.write_sample(sample)
                    self.reserved_cnt += 1
                else:
                    self.filtered_cnt += 1
            except Exception as e:
                print(traceback.format_exc())
                self.failed_cnt += 1

def main():
    parser = argparse.ArgumentParser(description="clean data")
    parser.add_argument("config_file")
    parser.add_argument(
        "--master-addr",
        type=str,
        required=True)
    args = parser.parse_args()
    setup_proc(args)
    cfg = OmegaConf.load(args.config_file)
    print('config', cfg)
    worker = Worker(cfg)
    worker.run()

if __name__ == '__main__':
    main()
