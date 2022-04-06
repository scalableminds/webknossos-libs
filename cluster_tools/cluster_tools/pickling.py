import os
import pickle
import sys

from .util import warn_after

WARNING_TIMEOUT = 10 * 60  # seconds


def file_path_to_absolute_module(file_path):
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


def get_suitable_pickle_protocol():
    # Protocol 4 allows to serialize objects larger than 4 GiB, but is only supported
    # beginning from Python 3.4
    protocol = 4 if sys.version_info[0] >= 3 and sys.version_info[1] >= 4 else 3
    return protocol


@warn_after("pickle.dumps", WARNING_TIMEOUT)
def dumps(*args, **kwargs):
    return pickle.dumps(*args, protocol=get_suitable_pickle_protocol(), **kwargs)


@warn_after("pickle.dump", WARNING_TIMEOUT)
def dump(*args, **kwargs):
    return pickle.dump(*args, protocol=get_suitable_pickle_protocol(), **kwargs)


@warn_after("pickle.loads", WARNING_TIMEOUT)
def loads(*args, **kwargs):
    assert (
        "custom_main_path" not in kwargs
    ), "loads does not implement support for the argument custom_main_path"
    return pickle.loads(*args, **kwargs)


class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == "__main__" and self.custom_main_path is not None:
            renamed_module = self.custom_main_path

        return super(RenameUnpickler, self).find_class(renamed_module, name)


@warn_after("pickle.load", WARNING_TIMEOUT)
def load(f, custom_main_path=None):
    unpickler = RenameUnpickler(f)
    unpickler.custom_main_path = (  # pylint: disable=attribute-defined-outside-init
        custom_main_path
    )
    return unpickler.load()
