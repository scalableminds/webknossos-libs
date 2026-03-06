# `upsample`

The `upsample` command upsamples a WEBKNOSSOS dataset. It works with both local datasets and datasets hosted on a WEBKNOSSOS server.

## Usage

```bash
webknossos upsample [OPTIONS] SOURCE
```

### Arguments

- **SOURCE**
    Path to a local WEBKNOSSOS dataset, or URL to a dataset on a WEBKNOSSOS server.
    Examples:
    - `/path/to/dataset`
    - `https://webknossos.org/datasets/Organization_X/my_dataset`

### Options

- `--from-mag`
    Magnification level to start upsampling from.
    Accepts a single number or a minus-separated string (e.g., `2` or `2-2-2`).
    Example: `--from-mag 2`.

- `--sampling-mode`
    The sampling mode to use for upsampling.
    Options: `anisotropic`, `isotropic`, `constant_z`.
    Default: `anisotropic`.

- `--layer-name`
    Name of the layer that should be upsampled.
    If not provided, all layers are processed.
    Example: `--layer-name gray_matter`.

#### WEBKNOSSOS context

- `--token`
    Authentication token for the WEBKNOSSOS instance (see https://webknossos.org/auth/token).
    Can also be provided via the `WK_TOKEN` environment variable.
    Required when SOURCE is a WEBKNOSSOS server URL pointing to a non-public dataset.

#### Executor options

- `--jobs`
    Number of processes to be spawned for parallel execution.
    Default: Number of CPU cores.
    Example: `--jobs 4`.

- `--distribution-strategy`
    Strategy to distribute the task across CPUs or nodes.
    Options: `multiprocessing`, `slurm`, `kubernetes`, `sequential`.
    Default: `multiprocessing`.

- `--job-resources`
    Specify resources for jobs when using the SLURM distribution strategy.
    Should be a JSON string.
    Example: `--job-resources '{"mem": "10M"}'`.

## Example Commands

### Upsample all layers of a local dataset:
```bash
webknossos upsample --from-mag 2 /path/to/dataset
```

### Upsample a specific layer:
```bash
webknossos upsample --from-mag 2 --layer-name gray_matter /path/to/dataset
```

### Upsample with parallel execution using SLURM:
```bash
webknossos upsample --from-mag 2 --distribution-strategy slurm --job-resources '{"mem": "10M"}' /path/to/dataset
```

### Upsample a dataset on a WEBKNOSSOS server:
```bash
webknossos upsample --from-mag 2 --token YOUR_TOKEN https://webknossos.org/datasets/Organization_X/my_dataset
```

### Upsample a specific layer of a remote dataset:
```bash
webknossos upsample --from-mag 2 --layer-name gray_matter --token YOUR_TOKEN https://webknossos.org/datasets/Organization_X/my_dataset
```

## Notes

- Local and remote datasets are distinguished automatically: a URL (starting with `http://` or `https://`) is treated as a remote dataset; any other path is treated as local.
- `--token` is only required when accessing non-public remote datasets. It can be set once via the `WK_TOKEN` environment variable.
- If `--layer-name` is omitted, all layers in the dataset are upsampled.
