import os
import os.path as osp
from . import BaseMonitor
from copy import deepcopy
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from recipes.ViT.metric import Metric


class ViTMonitor(BaseMonitor):

    def __init__(self, config, ctx, **kwargs):
        super().__init__(config, ctx, **kwargs)
        self.op_dict = dict()
        self.interim_metrics = dict()
        self.permanent_metrics = dict()
        self.metrics_names = list()

        self.global_step = kwargs.get("start_step", 0)
        self.rank = ctx.get('rank', 0)
        self.world_size = ctx.get('world_size', 1)
        self.local_rank = ctx.args.local_rank

        self.tb_writer = None

    def setup(self):
        config = self.config
        if self.rank == 0:
            self.tb_writer = SummaryWriter(log_dir=osp.join(config.output.dir, "log"))
        else:
            self.tb_writer = None
        self.strategy.setup()
        self.verbose.setup()

    def register_permanent_metric(self, **kwargs):
        assert "init_value" in kwargs
        name = kwargs.pop("name")
        assert name not in self.permanent_metrics, "duplicate metric names {}.".format(name)
        method = kwargs.pop("method", None) or "add"
        report_name = kwargs.pop("report_name", None) or name
        verbose_name = kwargs.pop("verbose_name", None) or name
        value = deepcopy(kwargs["init_value"])
        buffer = deepcopy(kwargs.get("init_buffer", dict()))
        self.metrics_names.append(name)
        self.permanent_metrics[name] = Metric(
            name=name,
            method=method,
            report_name=report_name,
            verbose_name=verbose_name,
            value=value,
            buffer=buffer,
            **kwargs
        )

    def register_interim_metrics(self, **kwargs):
        name = kwargs.pop("name")
        assert name not in self.permanent_metrics, "duplicate metric names {}.".format(name)
        method = kwargs.pop("method", None) or "assign"
        report_name = kwargs.pop("report_name", None) or name
        verbose_name = kwargs.pop("verbose_name", None) or name
        value = deepcopy(kwargs.get("init_value", 0))
        buffer = deepcopy(kwargs.get("init_buffer", dict()))
        self.metrics_names.append(name)
        self.interim_metrics[name] = Metric(
            name=name,
            method=method,
            report_name=report_name,
            verbose_name=verbose_name,
            value=value,
            buffer=buffer,
            **kwargs
        )

    def increment(self):
        self.global_step += 1

    def reset(self):
        for metric in self.interim_metrics.values():
            metric.reset()

    def report(self, name, data):
        tb_writer = self.tb_writer
        if tb_writer is not None:
            tb_writer.add_scalar(
                name,
                data,
                global_step=self.global_step,
                new_style=True
            )

    def collect(self):
        pass

    def print(self, *args, **kwargs):
        self.verbose.print(*args, **kwargs)

    def step(self, ctx):
        self.increment()
        for name in self.metrics_names:
            metric = self.interim_metrics.get(name, None) or self.permanent_metrics.get(name, None)
            assert metric is not None
            if not metric.enabled:
                continue
            metric.update(ctx)
            if self.global_step % metric.verbose_per_step == 0:
                self.print(metric.verbose_name, ":", metric.value, rank=0)
            if self.global_step % metric.report_per_step == 0:
                self.report(metric.verbose_name, metric.value)
        self.verbose.step()
        self.strategy.step()
        self.print("-" * 100, rank=0)
