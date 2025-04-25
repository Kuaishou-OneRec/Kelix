from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Callable, Any, Union

from torch._C._distributed_c10d import ProcessGroup
from torch.distributed.distributed_c10d import _get_default_group

__all__ = [
    "MetricsType",
    "MetricsItem",
    "DistMetricsItem",
    "disable_profiling",
    "profiling_enabled",
]


class MetricsType(Enum):
    SIMPLE = 1
    TIMER = 2
    COLLECTIVE = 3


class MetricsItem(ABC):
    def __init__(self, name: str, type: MetricsType):
        self.name = name
        self.type = type

    @abstractmethod
    def accum(self, other):
        assert False, "not implemented in base class"

    @abstractmethod
    def materialize(self):
        assert False, "not implemented in base class"

    @abstractmethod
    def table_schema(self):
        assert False, "not implemented in base class"

    @abstractmethod
    def table_data(self):
        assert False, "not implemented in base class"


class DistMetricsItem(ABC):
    def __init__(
        self,
        members: List[MetricsItem],
        pg: ProcessGroup,
    ) -> None:
        assert len(members) > 0, "empty dist metrics"
        self.name = members[0].name
        self.type = members[0].type
        self.members = members
        self.pg = pg

    @abstractmethod
    def materialize(self):
        assert False, "not implemented in base class"

    @abstractmethod
    def table_schema(self):
        assert False, "not implemented in base class"

    @abstractmethod
    def table_data(self):
        assert False, "not implemented in base class"


class TableData(object):
    def __init__(self, schema: List[str]):
        assert len(schema) > 0, "empty schema"
        assert schema[0] == "name", "schema's first item must be name"
        self.schema = schema
        self.rows: List[List[Any]] = []
        self.datas: Dict[str, Dict[str, Any]] = {}

    def add_row(self, row: List[Any]):
        assert len(row) == len(
            self.schema
        ), f"row[{row}] not match schema[{self.schema}]"
        self.rows.append(row)
        for i in range(1, len(row)):
            self.datas.setdefault(row[0], {})[self.schema[i]] = row[i]

    def empty(self):
        return not self.rows

    def get(self, key: str, *sub_keys: str) -> Any:
        datas = self.datas[key]
        for sk in sub_keys:
            if sk in datas:
                return datas[sk]

        return None

    def get_keys(self) -> List[str]:
        return list(self.datas.keys())


class TableFormater(object):
    def format(self, inputs: List[Union[MetricsItem, DistMetricsItem]]):
        assert len(inputs) > 0, f"empty input metrics"
        data = TableData(inputs[0].table_schema())
        for item in inputs:
            if hasattr(item, "pg"):
                pg = item.pg
            else:
                pg = _get_default_group()

            if pg.rank() == 0:
                data.add_row(item.table_data())

        return data


class ReporterBase(ABC):
    @abstractmethod
    def report(self, metrics_type: MetricsType, data: TableData):
        assert False, "not implemented in base class"

    def step(self): ...


_ENABLE_PROFILING = True


def disable_profiling():
    global _ENABLE_PROFILING
    _ENABLE_PROFILING = False


def profiling_enabled():
    global _ENABLE_PROFILING
    return _ENABLE_PROFILING
