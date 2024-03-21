import os
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from distributed import LocalCluster

import cluster_tools

_dask_cluster: Optional["LocalCluster"] = None


def job(_arg: None) -> str:
    return os.getcwd()


def test_pass_cwd() -> None:
    global _dask_cluster
    if not _dask_cluster:
        from distributed import LocalCluster, Worker

        _dask_cluster = LocalCluster(
            worker_class=Worker, resources={"mem": 20e9, "cpus": 4}, nthreads=6
        )
    with cluster_tools.get_executor(
        "dask", job_resources={"address": _dask_cluster}
    ) as executor:
        tmp_path = os.path.realpath("/tmp")  # macOS redirects `/tmp` to `/private/tmp`
        os.chdir(tmp_path)
        assert list(executor.map(job, [None])) == [tmp_path]
