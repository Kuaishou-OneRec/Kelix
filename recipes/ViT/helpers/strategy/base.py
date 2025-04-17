import os
import os.path as osp
from copy import deepcopy
from abc import ABC, abstractmethod


class BaseStrategy(object):

    def __init__(self, config, ctx, **kwargs):
        self.config = config
        self.ctx = ctx
        self.kwargs = kwargs
        self.seed = config.get("seed", 1234)

    def set_random_seed(self, seed=None):
        seed = seed or self.seed
        set_transformers_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    @abstractmethod
    def setup(self):
        raise NotImplementedError

    @abstractmethod
    def report(self, **kwargs):
        raise NotImplementedError
