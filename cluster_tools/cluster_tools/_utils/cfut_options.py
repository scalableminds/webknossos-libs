"""Handling of the `__cfut_options` kwarg and the `map_to_futures` getters, which
select where a job's output is persisted. Shared by all executors."""

import logging
import os
from collections.abc import Callable
from typing import Any, TypedDict, TypeVar, cast

from typing_extensions import NotRequired, ParamSpec

from cluster_tools._utils import pickling
from cluster_tools.output_store import OutputStore

_T = TypeVar("_T")
_P = ParamSpec("_P")
_S = TypeVar("_S")


# A job's output is persisted as the pickled tuple `(True, result)` in the executor's
# `OutputStore` (only in the success case, so that it can serve as a checkpoint).
# `output_key` selects where; for the default `FileOutputStore` it is a file path
# and `output_pickle_path` is accepted as an alias.
class CFutDict(TypedDict):
    output_key: NotRequired[str]
    output_pickle_path: NotRequired[str | os.PathLike]


def parse_cfut_options(kwargs: dict[str, Any]) -> str | None:
    """Removes `__cfut_options` from kwargs and returns the output key, if any."""
    if "__cfut_options" not in kwargs:
        return None
    options = cast(CFutDict, kwargs["__cfut_options"])
    del kwargs["__cfut_options"]
    output_key = options.get("output_key")
    output_pickle_path = options.get("output_pickle_path")
    if output_key is None and output_pickle_path is None:
        raise ValueError("__cfut_options must contain output_key.")
    if output_key is not None and output_pickle_path is not None:
        raise ValueError(
            "__cfut_options must not contain both output_key and output_pickle_path."
        )
    return output_key if output_key is not None else str(output_pickle_path)


def resolve_output_key_getter(
    output_pickle_path_getter: Callable[[_S], os.PathLike] | None,
    output_key_getter: Callable[[_S], str] | None,
) -> Callable[[_S], str] | None:
    """Merges the `map_to_futures` getters into a single key getter."""
    if output_key_getter is not None and output_pickle_path_getter is not None:
        raise ValueError(
            "Specify either output_key_getter or output_pickle_path_getter, not both."
        )
    if output_key_getter is not None:
        return output_key_getter
    if output_pickle_path_getter is not None:
        getter = output_pickle_path_getter
        return lambda arg: str(getter(arg))
    return None


def cfut_options_kwargs(
    arg: _S,
    output_pickle_path_getter: Callable[[_S], os.PathLike] | None,
    output_key_getter: Callable[[_S], str] | None,
) -> dict[str, CFutDict]:
    """Builds the `__cfut_options` kwarg for `submit` from the `map_to_futures` getters."""
    key_getter = resolve_output_key_getter(output_pickle_path_getter, output_key_getter)
    if key_getter is None:
        return {}
    return {"__cfut_options": {"output_key": key_getter(arg)}}


def execute_and_persist(
    output_store: OutputStore,
    output_key: str,
    fn: Callable[_P, _T],
    *args: _P.args,
    **kwargs: _P.kwargs,
) -> _T:
    """Runs `fn` in the job and stores its result. Used by the in-process executors;
    the cluster executors do the same in `cluster_tools.remote`."""
    try:
        result = fn(*args, **kwargs)
    except Exception as exc:
        logging.warning(f"Job computation failed with:\n{exc.__repr__()}")
        raise exc
    else:
        # Only store the result in the success case, since the output
        # is used as a checkpoint. The cluster executor also stores failures,
        # since it needs them to transport the exception back.
        output_store.write(output_key, pickling.dumps((True, result)), success=True)
        return result
