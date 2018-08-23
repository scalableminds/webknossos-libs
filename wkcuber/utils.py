import wkw
import numpy as np
from glob import iglob
from collections import namedtuple
from multiprocessing import cpu_count, Lock
from concurrent.futures import ProcessPoolExecutor
from os import path
from platform import python_version
from math import floor, ceil


from .knossos import KnossosDataset, CUBE_EDGE_LEN


WkwDatasetInfo = namedtuple(
    "WkwDatasetInfo", ("dataset_path", "layer_name", "dtype", "mag")
)
KnossosDatasetInfo = namedtuple("KnossosDatasetInfo", ("dataset_path", "dtype"))


def _open_wkw(info):
    return wkw.Dataset.open(
        path.join(info.dataset_path, info.layer_name, str(info.mag)),
        wkw.Header(np.dtype(info.dtype)),
    )


def open_wkw(info, lock=None):
    if lock is None:
        return _open_wkw(info)
    else:
        with lock:
            return _open_wkw(info)


def open_knossos(info):
    return KnossosDataset.open(info.dataset_path, np.dtype(info.dtype))


def add_verbose_flag(parser):
    parser.add_argument(
        "--verbose", "-v", help="Verbose output", dest="verbose", action="store_true"
    )

    parser.set_defaults(verbose=False)


def find_files(source_path, extensions):
    # Find all files with a matching file extension
    return (
        f
        for f in iglob(source_path, recursive=True)
        if any([f.endswith(suffix) for suffix in extensions])
    )


def get_chunks(arr, chunk_size):
    for i in range(0, len(arr), chunk_size):
        yield arr[i : i + chunk_size]


def get_regular_chunks(min_z, max_z, chunk_size):
    i = floor(min_z / chunk_size) * chunk_size
    while i < ceil(max_z / chunk_size) * chunk_size:
        yield range(i, i + chunk_size)
        i += chunk_size


def add_jobs_flag(parser):
    parser.add_argument("--jobs", "-j", help="Parallel jobs", default=cpu_count())


def pool_init(lock):
    global process_pool_lock
    process_pool_lock = lock


def pool_get_lock():
    global process_pool_lock
    try:
        return process_pool_lock
    except NameError:
        return None


class ParallelExecutor:
    def __init__(self, jobs):
        self.lock = Lock()
        if python_version() >= "3.7.0":
            self.exec = ProcessPoolExecutor(
                jobs, initializer=pool_init, initargs=(self.lock,)
            )
        else:
            self.exec = ProcessPoolExecutor(jobs)
        self.futures = []

    def submit(self, fn, *args):
        future = self.exec.submit(fn, *args)
        self.futures.append(future)
        return future

    def __enter__(self):
        self.exec.__enter__()
        return self

    def __exit__(self, type, value, tb):
        [f.result() for f in self.futures]
        self.exec.__exit__(type, value, tb)
