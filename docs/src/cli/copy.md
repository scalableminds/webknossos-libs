# `copy-dataset`

The `copy-dataset` command allows you to create a copy of a WEBKNOSSOS dataset. It supports copying datasets locally or from remote paths (e.g., HTTP or S3). The command provides options to customize the data format, chunking, sharding, and parallel execution.

## Usage

```bash
copy-dataset [OPTIONS] SOURCE TARGET
```

### Arguments

- **SOURCE**  
    Path to the source WEBKNOSSOS dataset.  
    Example: `/path/to/source/dataset` or `s3://bucket-name/source-dataset`.

- **TARGET**  
    Path to the target WEBKNOSSOS dataset.  
    Example: `/path/to/target/dataset` or `s3://bucket-name/target-dataset`.

### Options
- `--data-format`  
    Specify the data format to store the target dataset.  
    Options: `wkw`, `zarr`, `zarr3`  
    Example: `--data-format zarr3`.

- `--chunk-shape`  
    Number of voxels to be stored as a chunk in the target dataset.  
    Example: `--chunk-shape 32,32,32`.

- `--shard-shape`  
    Number of voxels to be stored as a shard in the target dataset.  
    Example: `--shard-shape 1024,1024,1024`.

- `--exists-ok`  
    Allow overwriting an existing dataset.  
    Default: `False`.

- `--jobs`  
    Number of processes to be spawned for parallel execution.  
    Default: Number of CPU cores.

- `--distribution-strategy`  
    Strategy to distribute the task across CPUs or nodes.  
    Options: `multiprocessing`, `slurm`, `kubernetes`, `sequential`.  
    Default: `multiprocessing`.

- `--job-resources`  
    Specify resources for jobs when using the SLURM distribution strategy.  
    Example: `--job-resources '{"mem": "10M"}'`.

### Environment Variables for Remote Paths

When using remote paths (e.g., HTTP or S3), configure the following environment variables:

#### HTTP Basic Authentication:
- `HTTP_BASIC_USER`  
- `HTTP_BASIC_PASSWORD`

#### S3 Configuration:
- `S3_ENDPOINT_URL`  
- `AWS_ACCESS_KEY_ID`  
- `AWS_SECRET_ACCESS_KEY`

## Example Commands

### Copy a dataset locally:
```bash
webknossos copy-dataset /path/to/source/dataset /path/to/target/dataset
```
This command copies a dataset from a local source path to a local target path.

### Copy a local dataset to a S3 storage:
```bash
AWS_ACCESS_KEY_ID=XXX AWS_SECRET_ACCESS_KEY=XXX \
webknossos copy-dataset \
  --data-format zarr3 \
  --jobs 4 \
  data/source s3://webknossos-bucket/target
```
This command copies a local dataset to a remote S3 bucket. Ensure that the required S3 environment variables are set.

### Copy a dataset with parallel execution:
```bash
webknossos copy-dataset --jobs 4 /path/to/source/dataset /path/to/target/dataset
```
This command uses 4 parallel processes to speed up the dataset copying process.

### Copy a dataset using SLURM with custom job resources:
```bash
webknossos copy-dataset --distribution-strategy slurm --job-resources '{"mem": "10M"}' /path/to/source/dataset /path/to/target/dataset
```
This command uses the SLURM distribution strategy with custom memory allocation for each job.

## Notes

- Ensure that the source and target paths are accessible and properly configured.
- Use the `--exists-ok` option to overwrite an existing target dataset if necessary.
- For remote paths, make sure the required environment variables are set.
- This command is designed to handle large datasets efficiently by leveraging parallel processing and customizable storage configurations.