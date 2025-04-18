from dataclasses import dataclass
from copy import deepcopy
from typing import Any, Union, Callable, Dict


@dataclass
class Metric:

    name: str = ""
    enabled: bool = True
    larger_is_better: bool = True
    method: Union[str, Callable] = "add"
    init_value: Any = None
    value: Any = None
    buffer: Dict[str, Any] = None
    init_buffer: Dict[str, Any] = None
    verbose_per_step: int = 1
    report_per_step: int = 1
    report_name: str = ""
    verbose_name: str = ""
    formula: str = ""
    reset_step: int = 0x3f3f3f3f

    def add_value(self, other: Dict[str, Any]):
        value = getattr(other, self.name)
        self.value = self.value + value

    def prod_value(self, other: Dict[str, Any]):
        value = getattr(other, self.name)
        self.value = self.value * value

    def minus_value(self, other: Dict[str, Any]):
        value = getattr(other, self.name)
        self.value = self.value - value

    def divide_value(self, other: Dict[str, Any]):
        value = getattr(other, self.name)
        self.value = self.value / value

    def assign_value(self, other: Dict[str, Any]):
        value = getattr(other, self.name)
        self.value = deepcopy(value)

    def update(self, other: Dict[str, Any]):
        if isinstance(self.method, str):
            method = self.method
            update_method = getattr(self, "{}_value".format(method))
            update_method(other)
        else:
            assert isinstance(self.method, Callable)
            update_method: Callable = self.method
            update_method(self, other)

    def reset(self):
        self.value = deepcopy(self.init_value)
        self.buffer = deepcopy(self.init_buffer)
