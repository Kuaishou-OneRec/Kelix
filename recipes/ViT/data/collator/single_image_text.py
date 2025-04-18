import torch


class SingleImageTextPairCollator(object):

    def __init__(self, monitor=None, **kwargs):
        self.monitor = monitor
        self.kwargs = kwargs
    
    def __call__(self, batch):
        if self.monitor is not None:
            raise NotImplementedError
        samples = dict()
        for sample in batch:
            for key in sample["json"]:
                if key not in samples:
                    samples[key] = list()
                content = sample["json"][key]
                if key == "images":
                    assert len(content) == 1, "Multi-Images not supported yet."
                samples[key].append(content[0])

        return samples
