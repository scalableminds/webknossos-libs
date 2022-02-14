import cluster_tools


def square(n):
    return n * n


def main():
    with cluster_tools.get_executor(
        "kubernetes",
        job_resources={
            "memory": "100M",
            "python_executable": "python",
            "image": "scalableminds/cluster_tools:latest",
            "node_selector": {"kubernetes.io/hostname": "asterix.scm.io"},
            "namespace": "cluster-tools",
        },
    ) as exec:
        print(list(exec.map(square, [n + 2 for n in range(2)])))


if __name__ == "__main__":
    main()
