import torch


class IdentityCollator(object):

    def __init__(self, monitor=None, **kwargs):
        self.monitor = monitor
        self.kwargs = kwargs
    
    def __call__(self, batch):
        if self.monitor is not None:
            raise NotImplementedError
        return batch[0]
