# `downsample`

The `downsample` command allows you to reduce the resolution of a WEBKNOSSOS dataset. It supports downsampling all layers or a specific layer and provides options to control the sampling mode, coarseness, and parallel execution.

## Usage

```bash
webknossos downsample [OPTIONS] TARGET
```

### Arguments

- **TARGET**  
    Path to your WEBKNOSSOS dataset.  
    Example: `/path/to/dataset`

### Options

- `--sampling-mode`  
    The sampling mode to use during downsampling.  
    Available options are:
    - `anisotropic` (default): Downsamples each axis based on their individual resolution.
    - `isotropic`: Downsamples all axes uniformly.
    - `constant_z`: Downsamples the x and y axes while keeping the z-axis resolution constant.
    
    Example: `--sampling-mode anisotropic`

- `--layer-name`  
    Name of the layer to downsample.  
    If not provided, all layers are downsampled.  
    Example: `--layer-name gray`

- `--coarsest-mag`  
    The magnification level at which to stop downsampling.  
    Should be a number or a dash-separated string (e.g., `2` or `2-2-2`).  
    Example: `--coarsest-mag 2`

- `--jobs`  
    Number of processes to be spawned for parallel execution.  
    Default: Number of CPU cores.  
    Example: `--jobs 4`

- `--distribution-strategy`  
    Strategy to distribute the task across CPUs or nodes.  
    Options: `multiprocessing`, `slurm`, `kubernetes`, `sequential`. 
    Default: `multiprocessing`  
    Example: `--distribution-strategy sequential`

- `--job-resources`  
    JSON string to specify resources for jobs when using the SLURM distribution strategy.  
    Example: `--job-resources '{"mem": "10M"}'`

## Example Commands

### Downsample all layers of a dataset:
```bash
webknossos downsample /path/to/dataset
```
This command downsamples all layers of the dataset located at `/path/to/dataset` using the default settings.

### Downsample a specific layer with a custom sampling mode:
```bash
webknossos downsample --layer-name gray --sampling-mode anisotropic /path/to/dataset
```
This command downsamples only the "gray" layer with the anisotropic sampling mode.

### Downsample with parallel execution and custom job settings:
```bash
webknossos downsample --jobs 4 --distribution-strategy slurm --job-resources '{"mem": "10M"}' /path/to/dataset
```
This command uses 4 processes and the SLURM distribution strategy with custom memory allocation for each job.

### Downsample until a specified coarsest magnification:
```bash
webknossos downsample --coarsest-mag 2 /path/to/dataset
```
This command downsamples the dataset until the magnification level 2 is reached.

## Notes

- Ensure that the target dataset path is accessible.
- Use the `--layer-name` option to downsample a specific layer; if omitted, all layers are processed.
- Customize parallel execution with the `--jobs`, `--distribution-strategy`, and `--job-resources` options to optimize performance.
- The sampling mode and coarsest magnification offer control over the downsampling process to suit various dataset requirements.