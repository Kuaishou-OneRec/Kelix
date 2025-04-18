import os
import os.path as osp
import argparse
from copy import deepcopy
from abc import ABC, abstractmethod
from recipes.ViT.helpers.verbose import build_verbose
from recipes.ViT.helpers.strategy import build_strategy


class BaseMonitor(object):

    def __init__(self, config, ctx, *args, **kwargs):
        self.strategy = build_strategy(config, ctx, **kwargs)
        self.verbose = build_verbose(config, ctx, **kwargs)
        self.config = config
        self.ctx = ctx
        self.kwargs = kwargs
        self.args = args

    @abstractmethod
    def report(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def collect(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def setup(self, **kwargs):
        raise NotImplementedError
