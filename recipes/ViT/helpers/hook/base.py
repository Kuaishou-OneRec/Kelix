import os
import os.path as osp
from abc import ABC, abstractmethod


class BaseHook(object):

    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self.args = args

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError
