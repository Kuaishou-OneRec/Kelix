import torch
import numpy as np
from .vision_packing import VisionPackingCollator


class PackingBuffer(object):

    def __init__(self, max_len, buffer_idx=-1):
        self.buffer_idx = buffer_idx

        self.max_len = max_len
        self.cur_len = 0
        self.samples = list()

    def reset(self):
        self.samples = list()

    def can_append(self, vision_length):
        return self.cur_len + vision_length <= self.max_len

    def append(self, sample):
        self.samples.append(sample)


class BatchVisionPackingCollator(object):

    def __init__(self, monitor=None, **kwargs):
        self.monitor = monitor
        self.kwargs = kwargs
        self.max_packing_length = kwargs["packing"].max_length
        self.vision_collator = VisionPackingCollator(monitor, **kwargs)
    
    def __call__(self, batch):
        if self.monitor is not None:
            raise NotImplementedError
        
        batch = list(batch)
        batch.sort(key=lambda sample: -sample["json"]["seqlen"])

        buffers = list()

        batchs = list()
        for sample in batch:
            seqlen = sample["json"]["seqlen"]
            can_append_buffers = [
                buffer
                for buffer in buffers
                if buffer.can_append(seqlen)
            ]
            if len(can_append_buffers):
                buffer = min(can_append_buffers, key=lambda s: s.cur_len)
            else:
                buffer = PackingBuffer(max_len=self.max_packing_length, buffer_idx=len(buffers))
                buffers.append(buffer)
            buffer.append(sample)

        max_length = -1
        padding_lengths = list()
        for buffer in buffers:
            batchs.append(self.vision_collator(buffer.samples))
            max_length = max(max_length, buffer.cur_len)
            padding_lengths.append(max_length - buffer.cur_len)

        samples = dict()
        for key in batchs[0].keys():
            if key in ["image_indices", "image_position_ids", "height_position_ids", "width_position_ids"]:
                padding = [_batch[key].new_full((_padding_len, ), -1) for _padding_len, _batch in zip(padding_lengths, batchs)]
                new_values = [torch.concat([_batch[key], _padding], dim=0) for _batch, _padding in zip(batchs, padding)]
                samples[key] = torch.stack(new_values, dim=0)
            elif key in ["pixel_values"]:
                padding_shape = [(_padding_len, ) + _batch[key].shape[1:] for _padding_len, _batch in zip(padding_lengths, batchs)]
                padding = [_batch[key].new_zeros(_padding_shape) for _batch, _padding_shape in zip(batchs, padding_shape)]
                new_values = [torch.concat([_batch[key], _padding], dim=0) for _batch, _padding in zip(batchs, padding)]
                samples[key] = torch.stack(new_values, dim=0)
            elif key in ["sample_indices"]:
                values_list = [torch.LongTensor(-1)]
                for _batch in batchs:
                    values_list.append(_batch[key] + values_list[-1].max().item() + 1)
                values_list.pop(0)
                padding = [_value.new_full((_padding_len, ), -1) for _padding_len, _value in zip(padding_lengths, values_list)]
                new_values = [torch.concat([_value, _padding], dim=0) for _value, _padding in zip(values_list, padding)]
                samples[key] = torch.stack(new_values, dim=0)
            elif key in ["image_attention_mask"]:
                pass
            elif key in ["cu_seqlens"]:
                samples[key] = list()
                for _batch in batchs:
                    cu_seqlens = _batch[key]
                    if cu_seqlens[-1] != max_length:
                        cu_seqlens.append(max_length)
                    samples[key].append(cu_seqlens)
            else:
                samples[key] = list()
                for _batch in batchs:
                    samples[key].extend(_batch[key])
            sample_indices = samples["sample_indices"]
            image_attention_mask = (sample_indices[:, None, :] == sample_indices[:, :, None]).long()
            samples["image_attention_mask"] = image_attention_mask[:, None, :, :]
        
        return samples
