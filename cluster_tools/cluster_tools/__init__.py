from typing import Any, Literal, overload

from cluster_tools.executor_protocol import Executor
from cluster_tools.executors.dask import DaskExecutor
from cluster_tools.executors.debug_sequential import DebugSequentialExecutor
from cluster_tools.executors.multiprocessing_ import MultiprocessingExecutor
from cluster_tools.executors.pickle_ import PickleExecutor
from cluster_tools.executors.sequential import SequentialExecutor
from cluster_tools.schedulers.cluster_executor import (
    ClusterExecutor,
    RemoteOutOfMemoryException,
    RemoteResourceLimitException,
    RemoteTimeLimitException,
)
from cluster_tools.schedulers.kube import KubernetesExecutor
from cluster_tools.schedulers.pbs import PBSExecutor
from cluster_tools.schedulers.slurm import SlurmExecutor

# For backwards-compatibility:
WrappedProcessPoolExecutor = MultiprocessingExecutor


def _noop() -> bool:
    return True


did_start_test_multiprocessing = False


def _test_valid_multiprocessing() -> None:
    msg = """
    ###############################################################
    An attempt has been made to start a new process before the
    current process has finished its bootstrapping phase.

    This probably means that you are not using fork to start your
    child processes and you have forgotten to use the proper idiom
    in the main module:

        if __name__ == '__main__':
            main()
            ...
    ###############################################################
    """

    with get_executor("multiprocessing") as executor:
        try:
            res_fut = executor.submit(_noop)
            assert res_fut.result() == True, msg
        except RuntimeError as exc:
            raise Exception(msg) from exc
        except EOFError as exc:
            raise Exception(msg) from exc


@overload
def get_executor(environment: Literal["slurm"], **kwargs: Any) -> SlurmExecutor:
    ...


@overload
def get_executor(environment: Literal["pbs"], **kwargs: Any) -> PBSExecutor:
    ...


@overload
def get_executor(
    environment: Literal["kubernetes"], **kwargs: Any
) -> KubernetesExecutor:
    ...


@overload
def get_executor(environment: Literal["dask"], **kwargs: Any) -> DaskExecutor:
    ...


@overload
def get_executor(
    environment: Literal["multiprocessing"], **kwargs: Any
) -> MultiprocessingExecutor:
    ...


@overload
def get_executor(
    environment: Literal["sequential"], **kwargs: Any
) -> SequentialExecutor:
    ...


@overload
def get_executor(
    environment: Literal["debug_sequential"], **kwargs: Any
) -> DebugSequentialExecutor:
    ...


@overload
def get_executor(
    environment: Literal["test_pickling"], **kwargs: Any
) -> PickleExecutor:
    ...


def get_executor(environment: str, **kwargs: Any) -> "Executor":
    if environment == "slurm":
        return SlurmExecutor(**kwargs)
    elif environment == "pbs":
        return PBSExecutor(**kwargs)
    elif environment == "kubernetes":
        return KubernetesExecutor(**kwargs)
    elif environment == "dask":
        if "client" in kwargs:
            return DaskExecutor(kwargs["client"])
        else:
            return DaskExecutor.from_kwargs(**kwargs)
    elif environment == "multiprocessing":
        global did_start_test_multiprocessing
        if not did_start_test_multiprocessing:
            did_start_test_multiprocessing = True
            _test_valid_multiprocessing()

        return MultiprocessingExecutor(**kwargs)
    elif environment == "sequential":
        return SequentialExecutor(**kwargs)
    elif environment == "debug_sequential":
        return DebugSequentialExecutor(**kwargs)
    elif environment == "test_pickling":
        return PickleExecutor(**kwargs)
    raise Exception("Unknown executor: {}".format(environment))
