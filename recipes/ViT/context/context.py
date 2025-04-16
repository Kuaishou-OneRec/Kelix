import os
import json
import os.path as osp
import torch
import torch.distributed as dist


class Context(dict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)

    def __getitem__(self, key):
        return super().__getitem__(key)

    def __setattr__(self, key, value):
        super().__setitem__(key, value)

    def __getattr__(self, key):
        return super().__getitem__(key)


class DistributedContext(Context):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self):
        self["world_size"] = dist.get_world_size()
        self["rank"] = dist.get_rank()
        if dist.is_available() and dist.is_initialized():
            self["is_dist"] = True
        else:
            self["is_dist"] = False
        return self
