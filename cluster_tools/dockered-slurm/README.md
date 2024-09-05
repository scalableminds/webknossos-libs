# Slurm Docker Cluster

**Note: Refer to cluster_tools/README.md for how to use this container in the context of testing the cluster_tools**

This is a multi-container Slurm cluster using `docker compose`.  The compose file
creates named volumes for persistent storage of MySQL data files as well as
Slurm state and log directories.

## Containers and Volumes

The compose file will run the following containers:

* mysql
* slurmdbd
* slurmctld
* c1 (slurmd)
* c2 (slurmd)

The compose file will create the following named volumes:

* etc_munge         ( -> /etc/munge     )
* etc_slurm         ( -> /etc/slurm     )
* slurm_jobdir      ( -> /data          )
* var_lib_mysql     ( -> /var/lib/mysql )
* var_log_slurm     ( -> /var/log/slurm )

## Starting the Cluster

Run `docker compose` to instantiate the cluster:

```console
$ docker compose up -d
```

> Note: If you encounter permission errors (`Failed to check keyfile "/etc/munge/munge.key": Permission denied`), follow the steps from the "Deleting the Cluster" section and run the previous command again.

## Register the Cluster with SlurmDBD

To register the cluster to the slurmdbd daemon, run the `register_cluster.sh`
script:

```console
$ ./register_cluster.sh
```

> Note: You may have to wait a few seconds for the cluster daemons to become
> ready before registering the cluster.  Otherwise, you may get an error such
> as **sacctmgr: error: Problem talking to the database: Connection refused**.
>
> You can check the status of the cluster by viewing the logs: `docker compose
> logs -f`

> Note: If you encounter an error that the daemon is not running (`Error response from daemon: Container <...> is not running`), the start of the containers was not successful. Check the logs using `docker compose logs -f` and revisit the last step.

## Accessing the Cluster

Use `docker exec` to run a bash shell on the controller container:

```console
$ docker exec -it slurmctld bash
```

From the shell, execute slurm commands, for example:

```console
[root@slurmctld /]# sinfo
PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
normal*      up 5-00:00:00      2   idle c[1-2]
```

## Submitting Jobs

The `slurm_jobdir` named volume is mounted on each Slurm container as `/data`.
Therefore, in order to see job output files while on the controller, change to
the `/data` directory when on the **slurmctld** container and then submit a job:

```console
[root@slurmctld /]# cd /data/
[root@slurmctld data]# sbatch --wrap="uptime"
Submitted batch job 2
[root@slurmctld data]# ls
slurm-2.out
```

## Stopping and Restarting the Cluster

```console
$ docker compose stop
```

```console
$ docker compose start
```

## Deleting the Cluster

To remove all containers and volumes, run:

```console
$ docker compose rm -sf
$ docker volume rm dockered-slurm_etc_munge dockered-slurm_etc_slurm dockered-slurm_slurm_jobdir dockered-slurm_var_lib_mysql dockered-slurm_var_log_slurm
```
