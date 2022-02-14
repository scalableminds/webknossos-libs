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

The cluster_tools automatically determine the slurm limit for maximum array job size and split up larger job batches into multiple smaller batches.
Also, the slurm limit for the maximum number of jobs which are allowed to be submitted by a user at the same time is honored by looking up the number of currently submitted jobs and only submitting new batches if they fit within the limit.

If you would like to configure these limits independently, you can do so by setting the `SLURM_MAX_ARRAY_SIZE` and `SLURM_MAX_SUBMIT_JOBS` environment variables.

## Dev Setup

```
cd dockered-slurm
docker-compose up -d
docker exec -it slurmctld bash
docker exec -it c1 bash
```

Tests can be executed with `poetry run pytest -s tests/test.py` after entering the container.
Linting can be run with `./lint.sh`.
Code formatting (black) can be run with `./format.sh`.

## Credits

Thanks to [sampsyo/clusterfutures](https://github.com/sampsyo/clusterfutures) for providing the slurm core abstraction and [giovtorres/slurm-docker-cluster](https://github.com/giovtorres/slurm-docker-cluster) for providing the slurm docker environment which we use for CI based testing.
