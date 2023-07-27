import os
from typing import Callable

WARNING_TIMEOUT = 10 * 60  # seconds


def file_path_to_absolute_module(file_path: str) -> str:
    """
    Given a file path, return an import path.
    :param file_path: A file path.
    :return:
    """
    assert os.path.exists(file_path)
    file_loc, _ = os.path.splitext(file_path)
    directory, module = os.path.split(file_loc)
    module_path = [module]
    while True:
        if os.path.exists(os.path.join(directory, "__init__.py")):
            directory, package = os.path.split(directory)
            module_path.append(package)
        else:
            break
    path = ".".join(module_path[::-1])
    return path


def get_function_name(fun: Callable) -> str:
    # When using functools.partial, __name__ does not exist
    try:
        return fun.__name__ if hasattr(fun, "__name__") else get_function_name(fun.func)  # type: ignore[attr-defined]
    except Exception:
        return "<unknown function>"
