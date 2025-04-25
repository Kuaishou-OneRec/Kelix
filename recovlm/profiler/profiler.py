from typing import List, Dict, Union

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import ProcessGroup
from torch.distributed.distributed_c10d import _get_default_group

from .base import (
    MetricsType,
    MetricsItem,
    DistMetricsItem,
    TableData,
    TableFormater,
    ReporterBase,
    profiling_enabled,
)

from .reporter import (
    StdoutReporter,
    GrafanaReporter,
)

from .simple import (
    SimpleMetricsItem,
    DistSimpleMetricsItem,
    simple_metrics,
)

from .collective import (
    CollectiveMetricsItem,
    DistCollectiveMetricsItem,
    collective_metrics,
)

from .timer import (
    TimerMetricsItem,
    DistTimerMetricsItem,
    timer_metrics,
    aggregate_dist_item,
)

__all__ = [
    "enable_dist_aggregation",
    "enable_report_to_grafana",
    "profiler",
    "step",
    "set_report_file_path",
    "add_reporter",
]


class Profiler(object):
    def __init__(self):
        self.metrics_collectors = {
            MetricsType.SIMPLE: simple_metrics(),
            MetricsType.TIMER: timer_metrics(),
            MetricsType.COLLECTIVE: collective_metrics(),
        }

        self.do_dist_aggregation: bool = False
        self.local_results: Dict[MetricsType, List[MetricsItem]] = dict()
        self.dist_results: Dict[MetricsType, List[DistMetricsItem]] = dict()
        self.materialized = False
        self.formater = TableFormater()

        self.reporters: List[ReporterBase] = [
            StdoutReporter(),
        ]

    def add_reporter(self, reporter: ReporterBase):
        self.reporters.append(reporter)

    def step(self):
        if not profiling_enabled():
            return

        self.report()
        for _, collector in self.metrics_collectors.items():
            collector.step()

        for r in self.reporters:
            r.step()
        self.local_results.clear()
        self.dist_results.clear()
        self.materialized = False

    def report(self):
        if self.materialized:
            self._report()
            return

        torch.cuda.synchronize()
        for t, collector in self.metrics_collectors.items():
            if t not in self.local_results:
                self.local_results[t] = []

            self.local_results[t].extend(collector.get_all())

        default_pg = _get_default_group()
        if self.do_dist_aggregation:
            inputs_by_pg: Dict[ProcessGroup, List[MetricsItem]] = {}
            for t, item_list in self.local_results.items():
                for item in item_list:
                    if isinstance(item, CollectiveMetricsItem):
                        _ = item.get_and_del_pg()

                    pg = default_pg
                    if pg not in inputs_by_pg:
                        inputs_by_pg[pg] = []

                    inputs_by_pg[pg].append(item)

            for pg, item in inputs_by_pg.items():
                outputs = [None] * pg.size() if pg.rank() == 0 else None
                dist.gather_object(item, object_gather_list=outputs, group=pg)
                if pg.rank() == 0:
                    for metrics_idx in range(len(outputs[0])):
                        metrics_item = outputs[0][metrics_idx]
                        if metrics_item.type not in self.dist_results:
                            self.dist_results[metrics_item.type] = []

                        dist_metrics = self._create_dist_metrics(
                            [outputs[rank][metrics_idx] for rank in range(pg.size())],
                            pg=pg,
                        )
                        dist_metrics.materialize()
                        self.dist_results[metrics_item.type].append(dist_metrics)

            self._post_process_dist_timer_metrics()

        self.materialized = True
        self._report()

    def _post_process_dist_timer_metrics(self):
        if not self.dist_results or MetricsType.TIMER not in self.dist_results:
            return

        grouped_metrics: Dict[str, DistTimerMetricsItem] = dict()
        idx_map: Dict[str, int] = dict()
        idx = 0
        for item in self.dist_results[MetricsType.TIMER]:
            prefix = item.name.split("^^")[0]
            if prefix not in grouped_metrics:
                grouped_metrics[prefix] = []
                idx_map[prefix] = idx
                idx += 1

            grouped_metrics[prefix].append(item)

        self.dist_results[MetricsType.TIMER].clear()
        self.dist_results[MetricsType.TIMER] = [None] * len(grouped_metrics)
        for name, items in grouped_metrics.items():
            self.dist_results[MetricsType.TIMER][idx_map[name]] = aggregate_dist_item(
                name, items
            )

    def _report(self):
        assert self.materialized, "must be called after materialize"
        metrics = self.dist_results if self.do_dist_aggregation else self.local_results

        table_datas: Dict[MetricsType, TableData] = {}
        for metrics_type, value in metrics.items():
            if value:
                table_datas[metrics_type] = self.formater.format(value)

        for reporter in self.reporters:
            for k, v in table_datas.items():
                reporter.report(k, v)

    def _create_dist_metrics(self, items: List[MetricsItem], pg: ProcessGroup):
        assert len(items) > 0, "metrics metrics group"
        item = items[0]
        if isinstance(item, SimpleMetricsItem):
            return DistSimpleMetricsItem(items, pg=pg)
        elif isinstance(item, TimerMetricsItem):
            return DistTimerMetricsItem(items, pg=pg)
        elif isinstance(item, CollectiveMetricsItem):
            return DistCollectiveMetricsItem(items, pg)
        else:
            assert False, f"unknown MetricsItem type[{type(item)}]"


_INSTANCE = Profiler()


def profiler():
    global _INSTANCE
    return _INSTANCE


def enable_dist_aggregation():
    profiler().do_dist_aggregation = True


def enable_report_to_grafana(batch_size: int):
    for reporter in profiler().reporters:
        if isinstance(reporter, GrafanaReporter):
            return
    profiler().add_reporter(GrafanaReporter(batch_size))


def set_report_file_path(path: str):
    profiler().reporters[0].set_output_path(path)


def add_reporter(reporter: ReporterBase) -> None:
    profiler().add_reporter(reporter)


def step():
    profiler().step()
