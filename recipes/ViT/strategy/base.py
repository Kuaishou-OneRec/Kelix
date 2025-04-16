import os
import os.path as osp
from copy import deepcopy
from abc import ABC, abstractmethod


class BaseStrategy(object):

    def __init__(self, config, ctx, **kwargs):
        self.config = config
        self.ctx = ctx
        self.kwargs = kwargs

    @abstractmethod
    def setup(self):
        raise NotImplementedError

    @abstractmethod
    def report(self, **kwargs):
        raise NotImplementedError
