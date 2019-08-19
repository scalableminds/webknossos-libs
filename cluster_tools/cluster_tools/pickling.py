import sys
import os
import io
import logging

use_cloudpickle = 'USE_CLOUDPICKLE' in os.environ

if use_cloudpickle:
    import cloudpickle
    pickle_strategy = cloudpickle
else:
    import pickle
    pickle_strategy = pickle
import importlib
from .util import warn_after

WARNING_TIMEOUT = 10 * 60 # seconds

def file_path_to_absolute_module(file_path):
    """
    Given a file path, return an import path.
    :param file_path: A file path.
    :return:
    """
    assert os.path.exists(file_path)
    file_loc, ext = os.path.splitext(file_path)
    assert ext in ('.py', '.pyc')
    directory, module = os.path.split(file_loc)
    module_path = [module]
    while True:
        if os.path.exists(os.path.join(directory, '__init__.py')):
            directory, package = os.path.split(directory)
            module_path.append(package)
        else:
            break
    path = '.'.join(module_path[::-1])
    return path

@warn_after("pickle.dumps", WARNING_TIMEOUT)
def dumps(*args, **kwargs):
    # Protocol 4 allows to serialize objects larger than 4 GiB, but is only supported
    # beginning from Python 3.4
    protocol = 4 if sys.version_info[0] >= 3 and sys.version_info[1] >= 4 else 3
    pickled = pickle_strategy.dumps(*args, protocol=protocol, **kwargs)
    return pickled


@warn_after("pickle.loads", WARNING_TIMEOUT)
def loads(*args, **kwargs):
    return pickle_strategy.loads(*args, **kwargs)
