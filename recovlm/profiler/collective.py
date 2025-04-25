import os
import sys
import time
import contextlib
from typing import Callable, Dict, List
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import _get_default_group
from torch._C._distributed_c10d import ProcessGroup

from .base import (
    MetricsType,
    MetricsItem,
    DistMetricsItem,
    profiling_enabled,
)

__all__ = [
    "collective_tracer",
    "collective_metrics",
    "CollectiveMetricsItem",
    "DistCollectiveMetricsItem",
    "enable_collective_sanity_check",
    "disable_collective_sanity_check",
]


class _MiscUtils(object):
    @staticmethod
    def raw_name(coll_name: str):
        return "raw_" + coll_name

    @staticmethod
    def calc_bandwidth_mbs(size_bytes, dur_ms, pg):
        return float(size_bytes / (dur_ms + 1e-6) / 1e3) * ((pg.size() - 1) / pg.size())

    @staticmethod
    def is_local_group(process_group):
        granks = dist.get_process_group_ranks(process_group)
        local_gpu_count = torch.cuda.device_count()
        node_id = None
        for grank in granks:
            if node_id is None:
                node_id = grank // local_gpu_count
            elif node_id != grank // local_gpu_count:
                return False

        return True


class CollectiveMetricsItem(MetricsItem):
    def __init__(
        self,
        name: str,
        coll_name: str,
        data_size: int,
        start_ev: torch.cuda.Event,
        end_ev: torch.cuda.Event,
        pg: ProcessGroup,
    ):
        super().__init__(name, MetricsType.COLLECTIVE)
        self.coll_name = coll_name
        self.data_size = data_size
        self.start_ev = start_ev
        self.end_ev = end_ev
        self.bandwidth_mbs = None
        self.pg = pg
        self.materialized = False
        self.accum_items = []

        # materialized outputs
        self.dur_ms = []
        self.total_dur_ms = 0
        self.total_data_size = 0
        self.count = 0

    def accum(self, other):
        self.accum_items.append(other)

    def materialize(self):
        if not self.materialized:
            items = [self] + self.accum_items
            for item in items:
                dur = item.start_ev.elapsed_time(item.end_ev)
                self.dur_ms.append(dur)
                self.total_dur_ms += dur
                self.total_data_size += item.data_size
                self.count += 1

            self.bandwidth_mbs = _MiscUtils.calc_bandwidth_mbs(
                self.total_data_size, self.total_dur_ms, self.pg
            )
            self.start_ev = None
            self.end_ev = None
            self.accum_items.clear()
            self.materialized = True

    def get_and_del_pg(self):
        pg = self.pg
        self.pg = None
        return pg

    def table_schema(self):
        return [
            "name",
            "type",
            "aggregation_type",
            "collective_op",
            "total_dur(ms)",
            "total_size(Byte)",
            "count",
            "mean_bandwidth",
        ]

    def table_data(self):
        return [
            self.name,
            self.type,
            "local",
            self.coll_name,
            self.total_dur_ms,
            self.total_data_size,
            self.count,
            self.bandwidth_mbs,
        ]


class DistCollectiveMetricsItem(DistMetricsItem):
    def __init__(self, members: List[CollectiveMetricsItem], pg: ProcessGroup = None):
        super().__init__(members, pg=pg)
        self.dur_ms = 0
        self.data_size = members[0].total_data_size
        self.coll_name = members[0].coll_name
        self.count = members[0].count
        self.bandwidth_mbs = None

    def materialize(self):
        for i in range(self.count):
            self.dur_ms += min([x.dur_ms[i] for x in self.members])
        self.band_width_mbs = _MiscUtils.calc_bandwidth_mbs(
            self.data_size, self.dur_ms, self.pg
        )

    def table_schema(self):
        return [
            "name",
            "type",
            "aggregation_type",
            "collective_op",
            "total_dur(ms)",
            "total_data_size(Byte)",
            "count",
            "mean_bandwidth(MB/s)",
        ]

    def table_data(self):
        return [
            self.name,
            self.type,
            "dist_aggregation",
            self.coll_name,
            self.dur_ms,
            self.data_size,
            self.count,
            self.band_width_mbs,
        ]


class CollectiveMetrics(object):
    def __init__(self):
        self.items: Dict[str, CollectiveMetricsItem] = dict()
        self.materialized: bool = False

    def add(self, item: CollectiveMetricsItem):
        name = item.name
        if name not in self.items:
            self.items[name] = item
        else:
            self.items[name].accum(item)

    def get_all(self):
        if not self.materialized:
            for item in self.items.values():
                item.materialize()

            self.materialized = True
        return self.items.values()

    def step(self):
        self.items.clear()
        self.materialized = False


_COLLECTIVE_METRICS = CollectiveMetrics()


def collective_metrics():
    global _COLLECTIVE_METRICS
    return _COLLECTIVE_METRICS


_TRACER = None


def tracer():
    global _TRACER
    return _TRACER


_SANITY_CHECK = False


def enable_collective_sanity_check():
    global _SANITY_CHECK
    _SANITY_CHECK = True


def disable_collective_sanity_check():
    global _SANITY_CHECK
    _SANITY_CHECK = False


def sanity_check():
    global _SANITY_CHECK
    return _SANITY_CHECK


"""
collective_tracer
"""


@contextlib.contextmanager
def collective_tracer(
    name: str,
    sync_before_trace: bool = False,
):
    if not profiling_enabled:
        yield
    else:
        try:
            global _TRACER
            assert _TRACER is None, "do not support recursive call"
            _TRACER = CollectiveTracer(name=name, sync_before_trace=sync_before_trace)
            _TRACER.patch()
            yield _TRACER
        finally:
            _TRACER.unpatch()
            _TRACER = None


_COLLECTIVES = [
    "all_reduce",
    "all_gather",
    "all_gather_into_tensor",
    "reduce_scatter",
    "reduce_scatter_tensor",
    "all_to_all",
    "all_to_all_single",
]


class CollectiveTracer(object):
    def __init__(self, name: str, sync_before_trace: bool = True):
        self.name = name
        self.sync_before_trace: bool = sync_before_trace

    def patch(self):
        global _COLLECTIVES
        for coll in _COLLECTIVES:
            setattr(dist, _MiscUtils.raw_name(coll), getattr(dist, coll))
            setattr(dist, coll, getattr(CollectiveTracer, coll))

    def unpatch(self):
        global _COLLECTIVES
        for coll in _COLLECTIVES:
            raw_coll = _MiscUtils.raw_name(coll)
            setattr(dist, coll, getattr(dist, raw_coll))
            delattr(dist, raw_coll)

    @staticmethod
    def _collective_wapper(coll_name, data_size, *args, **kwargs):
        coll_fn = getattr(dist, _MiscUtils.raw_name(coll_name))
        pg = kwargs.get("group", _get_default_group())
        if not pg:
            pg = _get_default_group()
        if sanity_check():
            CollectiveTracer._sanity_check(coll_name, data_size, pg)

        async_op = kwargs.get("async_op", False)
        if tracer().sync_before_trace:
            torch.cuda.synchronize()
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        start_ev.record()
        if not async_op:
            coll_fn(*args, **kwargs)
            end_ev.record()
            metrics_item = CollectiveMetricsItem(
                tracer().name, coll_name, data_size, start_ev, end_ev, pg
            )
            collective_metrics().add(metrics_item)
        else:
            handle = coll_fn(*args, **kwargs)
            handle.wait()
            end_ev.record()
            metrics_item = CollectiveMetricsItem(
                tracer().name, coll_name, data_size, start_ev, end_ev, pg
            )
            collective_metrics().add(metrics_item)
            return handle

    @staticmethod
    def _sanity_check(coll_name: str, data_size: int, pg: ProcessGroup):
        if coll_name in ["all_to_all", "all_to_all_single"]:
            return

        size_tensor = torch.tensor(data_size, dtype=torch.long).cuda()
        outputs = (
            [torch.empty_like(size_tensor) for _ in range(pg.size())]
            if pg.rank() == 0
            else None
        )
        dist.gather(size_tensor, outputs, group=pg)
        size = None
        if pg.rank() == 0:
            for idx in range(len(outputs)):
                item_size = outputs[idx].cpu().item()
                if size is None:
                    size = item_size
                elif size != item_size:
                    assert (
                        False
                    ), f"sanity check failed [data_size: {size} vs {item_size}] for collective[{coll_name}]"

    @staticmethod
    def all_reduce(tensor, **kwargs):
        data_size = tensor.numel() * tensor.element_size() * 2
        return CollectiveTracer._collective_wapper(
            "all_reduce", data_size, *[tensor], **kwargs
        )

    @staticmethod
    def all_gather(tensor_list, tensor, **kwargs):
        data_size = tensor.numel() * tensor.element_size() * len(tensor_list)
        return CollectiveTracer._collective_wapper(
            "all_gather", data_size, *[tensor_list, tensor], **kwargs
        )

    @staticmethod
    def all_gather_into_tensor(out_tensor, in_tensor, **kwargs):
        data_size = out_tensor.numel() * out_tensor.element_size()
        return CollectiveTracer._collective_wapper(
            "all_gather_into_tensor", data_size, *[out_tensor, in_tensor], **kwargs
        )

    @staticmethod
    def reduce_scatter(output, input_list, **kwargs):
        data_size = output.numel() * output.element_size() * len(input_list)
        return CollectiveTracer._collective_wapper(
            "reduce_scatter", data_size, *[output, input_list], **kwargs
        )

    @staticmethod
    def reduce_scatter_tensor(out_tensor, in_tensor, **kwargs):
        data_size = in_tensor.numel() * in_tensor.element_size()
        return CollectiveTracer._collective_wapper(
            "reduce_scatter_tensor", data_size, *[out_tensor, in_tensor], **kwargs
        )

    @staticmethod
    def all_to_all(outputs, inputs, **kwargs):
        assert len(outputs) > 0 and len(inputs) > 0, ""
        data_size = outputs[0].element_size() * outputs[0].numel() * len(outputs)
        return CollectiveTracer._collective_wapper(
            "all_to_all", data_size, *[outputs, inputs], **kwargs
        )

    @staticmethod
    def all_to_all_single(output, input, **kwargs):
        data_size = output.element_size() * output.numel()
        return CollectiveTracer._collective_wapper(
            "all_to_all_single", data_size, *[output, input], **kwargs
        )
