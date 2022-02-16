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
    with cluster_tools.get_executor(
        "kubernetes",
        job_resources={
            "memory": "100M",
            "python_executable": "python",
            "image": "scalableminds/cluster_tools:latest",
            "node_selector": {},
            "namespace": "cluster-tools",
            "mounts": [os.path.abspath(os.curdir())],
        },
        debug=True,
    ) as exec:
        print(exec.map(list_dir, [os.path.abspath(os.curdir())]))


if __name__ == "__main__":
    test_simple()
    test_mounts()
