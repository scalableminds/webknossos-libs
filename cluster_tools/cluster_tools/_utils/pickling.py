import pickle
import sys
from typing import Any, BinaryIO, Optional

from cluster_tools._utils.warning import warn_after

WARNING_TIMEOUT = 10 * 60  # seconds


def _get_suitable_pickle_protocol() -> int:
    # Protocol 4 allows to serialize objects larger than 4 GiB, but is only supported
    # beginning from Python 3.4
    protocol = 4 if sys.version_info[0] >= 3 and sys.version_info[1] >= 4 else 3
    return protocol


@warn_after("pickle.dumps", WARNING_TIMEOUT)
def dumps(*args: Any, **kwargs: Any) -> bytes:
    return pickle.dumps(*args, protocol=_get_suitable_pickle_protocol(), **kwargs)  # type: ignore[misc]


@warn_after("pickle.dump", WARNING_TIMEOUT)
def dump(*args: Any, **kwargs: Any) -> None:
    pickle.dump(*args, protocol=_get_suitable_pickle_protocol(), **kwargs)  # type: ignore[misc]


@warn_after("pickle.loads", WARNING_TIMEOUT)
def loads(*args: Any, **kwargs: Any) -> Any:
    assert (
        "custom_main_path" not in kwargs
    ), "loads does not implement support for the argument custom_main_path"
    return pickle.loads(*args, **kwargs)


class _RenameUnpickler(pickle.Unpickler):
    custom_main_path: Optional[str]

    def find_class(self, module: str, name: str) -> Any:
        renamed_module = module
        if module == "__main__" and self.custom_main_path is not None:
            renamed_module = self.custom_main_path

        return super(_RenameUnpickler, self).find_class(renamed_module, name)


@warn_after("pickle.load", WARNING_TIMEOUT)
def load(f: BinaryIO, custom_main_path: Optional[str] = None) -> Any:
    unpickler = _RenameUnpickler(f)
    unpickler.custom_main_path = custom_main_path
    return unpickler.load()
