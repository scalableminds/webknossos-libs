import os

import cluster_tools


def square(n):
    return n * n


def list_dir(path):
    return os.listdir(path)


def test_simple():
    with cluster_tools.get_executor(
        "kubernetes",
        job_resources={
            "memory": "100M",
            "python_executable": "python",
            "image": "scalableminds/cluster_tools:latest",
            "node_selector": {},
            "namespace": "cluster-tools",
        },
        debug=True,
    ) as exec:
        assert list(exec.map(square, [n + 2 for n in range(2)])) == [4, 9]


def test_mounts():
    parent_dir = os.path.abspath(os.path.join(os.pardir, os.curdir))
    with cluster_tools.get_executor(
        "kubernetes",
        job_resources={
            "memory": "100M",
            "python_executable": "python",
            "image": "scalableminds/cluster_tools:latest",
            "node_selector": {},
            "namespace": "cluster-tools",
            "mounts": [parent_dir],
        },
        debug=True,
    ) as exec:
        assert "cluster_tools" in list(exec.map(list_dir, [parent_dir]))[0]


if __name__ == "__main__":
    test_simple()
    test_mounts()
