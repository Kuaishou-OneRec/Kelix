import torch
import numpy as np


class VisionPackingCollator(object):

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
                if isinstance(content, str):
                    samples[key].append(content)
                elif isinstance(content, (list, tuple)):
                    if key == "images":
                        assert len(content) == 1, "Multi-Images not supported yet."
                        content = content[0]
                    if key == "texts":
                        assert len(content) == 1, "Multi-Texts not supported yet."
                        content = content[0]
                    samples[key].append(content)
                else:
                    samples[key].append(content)
        samples["image_indices"] = torch.concat(samples["image_indices"], dim=0)
        
        samples["image_position_ids"] = torch.concat(samples.pop("position_ids"), dim=0)
        samples["height_position_ids"] = torch.concat(samples["height_position_ids"], dim=0)
        samples["width_position_ids"] = torch.concat(samples["width_position_ids"], dim=0)
        samples["pixel_values"] = torch.concat(samples["pixel_values"], dim=0).unsqueeze(0)

        sample_indices = list()
        for idx, seqlen in enumerate(samples["seqlen"]):
            sample_indices.extend([idx for _ in range(seqlen)])
        sample_indices = torch.LongTensor(sample_indices)
        samples["sample_indices"] = sample_indices
        image_attention_mask = (sample_indices[None, :, None] == sample_indices[None, None, :]).long()
        samples["image_attention_mask"] = image_attention_mask[:, None, :, :]

        samples["seqlen"].insert(0, 0)
        samples["cu_seqlens"] = list(np.cumsum(samples.pop("seqlen")))

        return samples
