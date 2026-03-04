# `remote-downsample`

The `remote-downsample` command downsamples a dataset that already exists on a WEBKNOSSOS server. It mirrors the functionality of [`downsample`](downsample.md) but operates on a remote dataset without requiring a local copy.

## Usage

```bash
webknossos remote-downsample [OPTIONS] TARGET
```

### Arguments

- **TARGET**
    Dataset name or URL on the WEBKNOSSOS server.
    Example: `https://webknossos.org/datasets/Organization_X/my_dataset`

### Options

#### WEBKNOSSOS context

- `--webknossos-url`
    URL to the WEBKNOSSOS instance.
    Can also be provided via the `WK_URL` environment variable.

- `--token`
    Authentication token for the WEBKNOSSOS instance (see https://webknossos.org/auth/token).
    Can also be provided via the `WK_TOKEN` environment variable.

#### Downsampling

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

#### Executor

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

### Downsample all layers of a remote dataset:
```bash
webknossos remote-downsample https://webknossos.org/datasets/Organization_X/my_dataset
```
Downsamples all layers of the dataset using default settings.

### Downsample up to a specific coarsest magnification:
```bash
webknossos remote-downsample --coarsest-mag 2 https://webknossos.org/datasets/Organization_X/my_dataset
```
Downsamples the dataset until magnification level 2 is reached.

### Downsample a specific layer with authentication:
```bash
webknossos remote-downsample \
    --webknossos-url https://webknossos.example.com \
    --token YOUR_TOKEN \
    --layer-name gray \
    --sampling-mode anisotropic \
    https://webknossos.example.com/datasets/Organization_X/my_dataset
```

### Downsample with parallel execution:
```bash
webknossos remote-downsample --jobs 4 --distribution-strategy slurm --job-resources '{"mem": "10M"}' https://webknossos.org/datasets/Organization_X/my_dataset
```

## Notes

- Authentication details can be provided via `--token` or the `WK_TOKEN` environment variable.
- Use `--layer-name` to downsample a specific layer; if omitted, all layers are processed.
- Customize parallel execution with `--jobs`, `--distribution-strategy`, and `--job-resources` to optimize performance.
