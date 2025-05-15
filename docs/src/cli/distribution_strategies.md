# Distribution Strategies of the CLI

The CLI offers several distribution strategies to execute tasks, each designed for different execution environments. You can choose one of the following:

## SLURM

Deploy tasks on a SLURM cluster to leverage distributed computing resources. When using the `slurm` strategy, you can configure SLURM-specific options to fine-tune your job scheduling and resource allocation. For example, you may supply job resources via parameters like:

```bash
--job-resources='{"mem": "10M", "cpus_per_task": "2", "ntasks": "4", "partition": "debug", "time": "00:05:00"}'
```

Below is a list of the allowed resources and a brief explanation of each. For further details on their meanings, please refer to the [SLURM sbatch documentation](https://slurm.schedmd.com/sbatch.html):

- **constraint**: Specifies job constraints (similar to the `-C, --constraint` option) to limit the job to nodes that match given characteristics.
- **cpus_per_task**: Sets the number of CPUs allocated per task (matches `--cpus-per-task`).
- **exclude**: Lists nodes to be excluded from job scheduling.
- **gres**: Requests generic resources (e.g., GPUs) with the same semantics as the `--gres` option.
- **mem**: Specifies the memory allocation for a job, equivalent to the memory parameter in SLURM.
- **nice**: Defines the scheduling priority (nice value) at the time of job submission.
- **ntasks**: Sets the total number of tasks for the job.
- **ntasks_per_node**: Specifies the number of tasks to run on each node (corresponds to `--ntasks-per-node`).
- **nodelist**: Provides an explicit list of nodes to use for job execution.
- **partition**: Designates the partition for the resource allocation (similar to `-p, --partition`).
- **reservation**: Indicates a reservation name if specific reserved resources need to be used.
- **time**: Sets the time limit for the job execution, as defined by SLURM.

## KUBERNETES

The `kubernetes` strategy allows tasks to execute within a Kubernetes cluster. It is ideal for containerized workflows and scalable orchestration. This strategy handles job creation and resource allocation automatically based on your Kubernetes configuration.

## MULTIPROCESSING

With the `multiprocessing` strategy, tasks are executed locally using parallel processes. This is useful when running on a single machine, allowing you to manage the number of concurrent jobs using the `--jobs` option. It simplifies debugging and iteration without requiring a cluster environment.

## SEQUENTIAL

The `sequential` strategy runs tasks one after the other. This strategy is suited for cases where parallel execution is not required or when tasks must be performed in a specific order.

Most tasks can be configured to be executed in a parallelized manner by choosing the appropriate distribution strategy for your environment. Use `--help` to get more information on configuring these strategies.