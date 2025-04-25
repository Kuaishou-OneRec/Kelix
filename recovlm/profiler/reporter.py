from prettytable import PrettyTable
from ..extension import core

from .base import ReporterBase, TableData, MetricsType

__all__ = ["StdoutReporter", "GrafanaReporter"]


class StdoutReporter(ReporterBase):
    def __init__(self):
        self.output_file = None

    def __del__(self):
        if self.output_file:
            self.output_file.close()

    def set_output_path(self, path: str):
        if self.output_file:
            self.output_file.close()

        self.output_file = open(path, "w")

    def report(self, metrics_type: MetricsType, data: TableData):
        if data.empty():
            return

        pt = PrettyTable()
        pt.field_names = data.schema
        pt.float_format = "0.2"
        pt.align[pt.field_names[0]] = "l"
        for row in data.rows:
            pt.add_row(row)

        if not self.output_file:
            print(pt)
        else:
            self.output_file.write(str(pt))
            self.output_file.flush()


class GrafanaReporter(ReporterBase):
    def __init__(self, batch_size):
        self._batch_size = batch_size
        super().__init__()

    def report_metric(self, data, index_name, metric_type="", scale=1, suffix=""):
        for row in data.rows:
            core.report_status_data(
                int(scale * row[data.schema.index(index_name)]),
                metric_type,
                row[data.schema.index("name")] + suffix,
                "",
            )

    def report_metric_block_time(
        self, data, index_name, metric_type="time_metrics", scale=1, suffix=""
    ):
        for row in data.rows:
            core.report_status_data(
                int(scale * row[data.schema.index(index_name)] / self._batch_size),
                metric_type,
                row[data.schema.index("name")] + suffix,
                "",
            )

    def report(self, metrics_type: MetricsType, data: TableData):
        if data.empty():
            return

        if metrics_type == MetricsType.SIMPLE:
            self.report_metric(data, "mean", "simple_metrics")
        elif metrics_type == MetricsType.TIMER:
            self.report_metric_block_time(data, "mean(ms)", scale=1000)
        elif metrics_type == MetricsType.COLLECTIVE:
            self.report_metric(
                data, "total_dur(ms)", "collective_metrics", 1000, "_time"
            )
            self.report_metric(
                data, "total_data_size(Byte)", "collective_metrics", 1000, "_data_size"
            )
            self.report_metric(
                data, "mean_bandwidth(MB/s)", "collective_metrics", 1000, "_bandwidth"
            )
        else:
            raise ValueError(f"Unknown metrics type: {metrics_type}")
