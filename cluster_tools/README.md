# Cluster Tools

[![CircleCI](https://circleci.com/gh/scalableminds/cluster_tools/tree/master.svg?style=svg)](https://circleci.com/gh/scalableminds/cluster_tools/tree/master)

This package provides python `Executor` classes for distributing tasks on a slurm cluster or via multi processing.

## Example

```python
import cluster_tools

def square(n):
  return n * n

if __name__ == '__main__':
  strategy = "slurm"  # other valid values are "multiprocessing" and "sequential"
  with cluster_tools.get_executor(strategy) as executor:
    result = list(executor.map(square, [2, 3, 4]))
    assert result == [4, 9, 16]
```

## Configuration

### Slurm

The `cluster_tools` automatically determine the slurm limit for maximum array job size and split up larger job batches into multiple smaller batches.
Also, the slurm limit for the maximum number of jobs which are allowed to be submitted by a user at the same time is honored by looking up the number of currently submitted jobs and only submitting new batches if they fit within the limit.

If you would like to configure these limits independently, you can do so by setting the `SLURM_MAX_ARRAY_SIZE` and `SLURM_MAX_SUBMIT_JOBS` environment variables.

### Kubernetes

#### Resource configuration

| Key                 | Description                                                                                                                                                                                                                                                              | Example                                 |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------- |
| `namespace`         | Kubernetes namespace for the resources to be created. Will be created if not existent.                                                                                                                                                                                   | `cluster-tools`                         |
| `node_selector`     | Which nodes to utilize for the processing. Needs to be a [Kubernetes `nodeSelector` object](https://kubernetes.io/docs/concepts/scheduling-eviction/assign-pod-node/).                                                                                                   | `{"kubernetes.io/hostname": "node001"}` |
| `image`             | The docker image for the containerized jobs to run in. The image needs to have the same version of `cluster_tools` and the code to run installed and in the `PYTHONPATH`.                                                                                                | `scalableminds/voxelytics:latest`       |
| `mounts`            | Additional mounts for the containerized jobs. The current working directory and the `.cfut` directory are automatically mounted.                                                                                                                                         | `["/srv", "/data"]`                     |
| `cpu`               | [CPU requirements](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/) for this job.                                                                                                                                                         | `4`                                     |
| `memory`            | [Memory requirements](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/) for this job. Not required, but highly recommended to avoid congestion. Without resource requirements, all jobs will be run in parallel and RAM will run out soon. | `16G`                                   |
| `python_executable` | The python executable may differ in the docker image from the one in the current environment. For images based of `FROM python`, it should be `python`. Defaults to `python`.                                                                                            | `python3.8`                             |
| `umask`             | `umask` for the jobs.                                                                                                                                                                                                                                                    | `0002`                                  |

#### Notes

- The jobs are run with the current `uid:gid`.
- The jobs are removed 7 days after completion (successful or not).
- The logs are stored in the `.cfut` directory. This is actually redundant, because Kubernetes also stores them.
- Pods are not restarted upon error.
- Requires Kubernetes ≥ 1.23.
- [Kubernetes cluster configuration](https://kubernetes.io/docs/concepts/configuration/organize-cluster-access-kubeconfig/) is expected to be the same as for `kubectl`, i.e. in `~/.kube/config` or similar.

## Dev Setup

```
cd dockered-slurm
docker-compose up -d
docker exec -it slurmctld bash
docker exec -it c1 bash
```

Tests can be executed with `cd tests && poetry run pytest -s tests.py` after entering the container.
Linting can be run with `./lint.sh`.
Code formatting (black) can be run with `./format.sh`.

## Credits

Thanks to [sampsyo/clusterfutures](https://github.com/sampsyo/clusterfutures) for providing the slurm core abstraction and [giovtorres/slurm-docker-cluster](https://github.com/giovtorres/slurm-docker-cluster) for providing the slurm docker environment which we use for CI based testing.
