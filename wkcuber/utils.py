import wkw
import numpy as np
from collections import namedtuple
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from os import path

from .knossos import KnossosDataset, CUBE_EDGE_LEN


WkwDatasetInfo = namedtuple(
    'WkwDatasetInfo', ('dataset_path', 'layer_name', 'dtype', 'mag'))
KnossosDatasetInfo = namedtuple(
    'KnossosDatasetInfo', ('dataset_path', 'layer_name', 'dtype', 'mag'))


def open_wkw(info):
    return wkw.Dataset.open(
        path.join(info.dataset_path, info.layer_name, str(info.mag)),
        wkw.Header(np.dtype(info.dtype)))


def open_knossos(info):
    return KnossosDataset.open(
        path.join(info.dataset_path, info.layer_name, str(info.mag)), 
        np.dtype(info.dtype))


def add_verbose_flag(parser):
    parser.add_argument(
        '--verbose', '-v',
        help="Verbose output",
        dest="verbose",
        action='store_true')

    parser.set_defaults(verbose=False)


def get_chunks(arr, chunk_size):
    for i in range(0, len(arr), chunk_size):
        yield arr[i:i + chunk_size]


def add_jobs_flag(parser):
    parser.add_argument(
        '--jobs', '-j',
        help="Parallel jobs",
        default=cpu_count())


class ParallelExecutor():
    def __init__(self, jobs):
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
