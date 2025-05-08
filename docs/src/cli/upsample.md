# `upsample`

The `upsample` command allows you to upsample a WEBKNOSSOS dataset. It supports upsampling either a specific layer or all layers in the dataset. The command provides options to customize the sampling mode, starting magnification, parallel execution, and distribution strategy.

## Usage

```bash
upsample [OPTIONS] SOURCE
```

### Arguments

- **SOURCE**  
    Path to your WEBKNOSSOS dataset.  
    Example: `/path/to/dataset` or `s3://bucket-name/dataset`.

### Options

- `--sampling-mode`  
    The sampling mode to use for upsampling.  
    Options: `anisotropic`, `isotropic`, `constant_z`.  
    Default: `anisotropic`. 

- `--from-mag`  
    Magnification level to start upsampling from.  
    Accepts a single number or a minus separated string (e.g., `2` or `2-2-2`).  
    Example: `--from-mag=2`.

- `--layer-name`  
    Name of the layer that should be upsampled. If not provided, all layers will be processed.  
    Example: `--layer-name=gray_matter`.

- `--jobs`  
    Number of processes to be spawned for parallel execution.  
    Default: Number of CPU cores.  
    Example: `--jobs=4`.

- `--distribution-strategy`  
    Strategy to distribute the task across CPUs or nodes.  
    Options: `multiprocessing`, `slurm`, `kubernetes`, `sequential`.  
    Default: `multiprocessing`. 

- `--job-resources`  
    Specify resources for jobs when using the SLURM distribution strategy.  
    Should be a JSON string.  
    Example: `--job-resources='{"mem": "10M"}'`.

## Example Commands

### Upsample all layers:
```bash
webknossos upsample --from-mag 2 /path/to/dataset
```
This command upsamples all layers in the dataset starting from magnification 2 using the default sampling mode.

### Upsample a specific layer:
```bash
webknossos upsample --from-mag 2-2-1 --layer-name gray_matter /path/to/dataset
```
This command upsamples only the layer named "gray_matter" in the dataset.

### Upsample with parallel execution using SLURM:
```bash
webknossos upsample --from-mag 2 --distribution-strategy slurm --job-resources='{"mem": "10M"}' /path/to/dataset
```
This command uses the SLURM distribution strategy with custom job resources for the upsampling process.

## Notes

- Ensure that the dataset path is accessible and properly configured.
- If `--layer-name` is omitted, the command will upsample all layers in the dataset.
- Adjust the `--jobs` option to optimize performance based on your system's resources.
- When using the SLURM strategy, make sure that the provided `--job-resources` JSON string correctly specifies the required resources.
