# `compress`

The `compress` command allows you to compress a WEBKNOSSOS dataset. It supports compressing specific layers and magnifications (mags) while leveraging parallel execution and customizable distribution strategies.

## Usage

```bash
webknossos compress [OPTIONS] TARGET
```

### Arguments

- **TARGET**  
    Path to the WEBKNOSSOS dataset to be compressed.  
    Example: `/path/to/dataset`.

### Options

- `--layer-name`  
    Name of the layer to be compressed. If not provided, all layers will be compressed.  
    Example: `--layer-name segmentation`.

- `--mag`  
    Magnifications (mags) to be compressed. Specify as a number or a hyphen-separated string (e.g., `2` or `2-2-2`).  
    For multiple mags, use the option multiple times.  
    Example: `--mag 1 --mag 2 --mag 4-4-2`.

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

## Example Commands

### Compress all layers in a dataset:
```bash
compress /path/to/dataset
```
This command compresses all layers in the specified dataset.

### Compress a specific layer:
```bash
compress --layer-name segmentation /path/to/dataset
```
This command compresses only the `segmentation` layer in the dataset.

### Compress specific magnifications:
```bash
compress --mag 1 --mag 2 /path/to/dataset
```
This command compresses the `1` and `2` magnifications in all layers of the dataset.

### Compress using SLURM with custom job resources:
```bash
compress --distribution-strategy slurm --job-resources '{"mem": "10M"}' /path/to/dataset
```
This command uses the SLURM distribution strategy with custom memory allocation for each job.

## Notes

- Ensure that the target dataset path is accessible and properly configured.
- Use the `--layer-name` option to target specific layers for compression.
- For SLURM distribution, make sure the required resources are specified using the `--job-resources` option.
- This command is designed to handle large datasets efficiently by leveraging parallel processing and customizable distribution strategies.
