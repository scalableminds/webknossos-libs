from concurrent import futures
from concurrent.futures import ProcessPoolExecutor
import os
import sys
import threading
from .schedulers.slurm import SlurmExecutor
from .schedulers.pbs import PBSExecutor
from .util import random_string, call
from . import pickling
import importlib

def get_existent_kwargs_subset(whitelist, kwargs):
    new_kwargs = {}
    for arg_name in whitelist:
        if arg_name in kwargs:
            new_kwargs[arg_name] = kwargs[arg_name]

    return new_kwargs


PROCESS_POOL_KWARGS_WHITELIST = ["max_workers", "mp_context", "initializer", "initargs"]
class WrappedProcessPoolExecutor(ProcessPoolExecutor):
    def __init__(self, **kwargs):
        new_kwargs = get_existent_kwargs_subset(PROCESS_POOL_KWARGS_WHITELIST, kwargs)

        ProcessPoolExecutor.__init__(self, **new_kwargs)


    def map_unordered(self, func, args):

        futs = self.map_to_futures(func, args)

        # Return a separate generator to avoid that map_unordered
        # is executed lazily (otherwise, jobs would be submitted
        # lazily, as well).
        def result_generator():
            for fut in futures.as_completed(futs):
                yield fut.result()

        return result_generator()

    def map_to_futures(self, func, args):

        futs = [self.submit(func, arg) for arg in args]
        return futs


class SequentialExecutor(WrappedProcessPoolExecutor):
    def __init__(self, **kwargs):
        kwargs["max_workers"] = 1
        WrappedProcessPoolExecutor.__init__(self, **kwargs)

def pickle_identity(obj):
    return pickling.loads(pickling.dumps(obj, True))

def pickle_identity_executor(func, *args, **kwargs):
    result = func(*args, **kwargs)
    return pickle_identity(result)

class PickleExecutor(WrappedProcessPoolExecutor):

    def submit(self, _func, *_args, **_kwargs):

        (func, args, kwargs) = pickle_identity((_func, _args, _kwargs))
        return super().submit(pickle_identity_executor, func, *args, **kwargs)


def get_executor(environment, **kwargs):
    if environment == "slurm":
        return SlurmExecutor(**kwargs)
    elif environment == "pbs":
        return PBSExecutor(**kwargs)
    elif environment == "multiprocessing":
        return WrappedProcessPoolExecutor(**kwargs)
    elif environment == "sequential":
        return SequentialExecutor(**kwargs)
    elif environment == "test_pickling":
        return PickleExecutor(**kwargs)
    raise Exception("Unknown executor: {}".format(environment))