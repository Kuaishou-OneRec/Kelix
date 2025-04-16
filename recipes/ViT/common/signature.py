import inspect
from copy import deepcopy


def filter_function_arguments(func, kwargs: dict, new_obj=True, exclude_keys=None):
    exclude_keys = exclude_keys or list()
    if new_obj:
        kwargs = deepcopy(kwargs)
    sig = inspect.signature(func)
    found = False
    for param in sig.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            found = True
            break
    if not found:
        return {
            key: value
            for key, value in kwargs.items()
            if key in sig.parameters and key not in exclude_keys
        }
    return {
        key: value
        for key, value in kwargs.items()
        if key not in exclude_keys
    }

