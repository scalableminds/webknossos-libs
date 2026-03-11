# `downsample`

The `downsample` command reduces the resolution of a WEBKNOSSOS dataset. It works with both local datasets and datasets hosted on a WEBKNOSSOS server.

## Usage

```bash
webknossos downsample [OPTIONS] TARGET
```

### Arguments

- **TARGET**
    Path to a local WEBKNOSSOS dataset, or URL to a dataset on a WEBKNOSSOS server.
    Examples:
    - `/path/to/dataset`
    - `https://webknossos.org/datasets/Organization_X/my_dataset`

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

#### WEBKNOSSOS context

- `--token`
    Authentication token for the WEBKNOSSOS instance (see https://webknossos.org/auth/token).
    Can also be provided via the `WK_TOKEN` environment variable.
    Required when TARGET is a WEBKNOSSOS server URL pointing to a non-public dataset.

- `--transfer-mode`
    Required for remote datasets. The transfer mode to use. Available options are `copy`, `move+symlink`, `symlink`, `http`.
    `copy`, `move+symlink`, `symlink` require direct filesystem access to the WEBKNOSSOS datastore.

- `--access-mode`
    How to access the remote dataset's data. Available options are `direct_path`, `zarr_streaming`, `proxy_path`.
    Defaults to `direct_path` when `--transfer-mode` is not `http`, otherwise `proxy_path`.

#### Executor options

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

### Downsample all layers of a local dataset:
```bash
webknossos downsample /path/to/dataset
```

### Downsample a specific layer with a custom sampling mode:
```bash
webknossos downsample --layer-name gray --sampling-mode anisotropic /path/to/dataset
```

### Downsample until a specified coarsest magnification:
```bash
webknossos downsample --coarsest-mag 2 /path/to/dataset
```

### Downsample with parallel execution and custom job settings:
```bash
webknossos downsample --jobs 4 --distribution-strategy slurm --job-resources '{"mem": "10M"}' /path/to/dataset
```

### Downsample a dataset on a WEBKNOSSOS server:
```bash
webknossos downsample --token YOUR_TOKEN --transfer-mode copy https://webknossos.org/datasets/Organization_X/my_dataset
```

### Downsample a specific layer of a remote dataset:
```bash
webknossos downsample --token YOUR_TOKEN --transfer-mode copy --layer-name gray https://webknossos.org/datasets/Organization_X/my_dataset
```

## Notes

- Local and remote datasets are distinguished automatically based on the TARGET: a URL (starting with `http://` or `https://`) is treated as a remote dataset; any other path is treated as local.
- `--token` is only required when accessing non-public remote datasets. It can be set once via the `WK_TOKEN` environment variable.
- `--transfer-mode` is required for remote datasets and ignored for local datasets.
- Use `--layer-name` to downsample a specific layer; if omitted, all layers are processed.
- Customize parallel execution with `--jobs`, `--distribution-strategy`, and `--job-resources` to optimize performance.
