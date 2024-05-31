import os
from typing import List

import cluster_tools


def square(n: float) -> float:
    return n * n


def list_dir(path: str) -> List[str]:
    return os.listdir(path)


def test_simple() -> None:
    with cluster_tools.get_executor(
        "kubernetes",
        job_resources={
            "memory": "100M",
            "python_executable": "python",
            "image": "scalableminds/cluster-tools:latest",
            "node_selector": {},
            "namespace": "cluster-tools",
        },
        debug=True,
    ) as executor:
        assert list(executor.map(square, [n + 2 for n in range(2)])) == [4, 9]


def test_mounts() -> None:
    parent_dir = os.path.abspath(os.path.join(os.pardir, os.curdir))
    with cluster_tools.get_executor(
        "kubernetes",
        job_resources={
            "memory": "100M",
            "python_executable": "python",
            "image": "scalableminds/cluster-tools:latest",
            "node_selector": {},
            "namespace": "cluster-tools",
            "mounts": [parent_dir],
        },
        debug=True,
    ) as executor:
        assert "cluster_tools" in list(executor.map(list_dir, [parent_dir]))[0]

    with cluster_tools.get_executor(
        "kubernetes",
        job_resources={
            "memory": "100M",
            "python_executable": "python",
            "image": "scalableminds/cluster-tools:latest",
            "node_selector": {},
            "namespace": "cluster-tools",
        },
        debug=True,
    ) as executor:
        assert "cluster_tools" not in list(executor.map(list_dir, [parent_dir]))[0]
