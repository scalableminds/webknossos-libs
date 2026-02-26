# `remote-convert`

The `remote-convert` command converts an image stack to a WEBKNOSSOS dataset and uploads it directly to a WEBKNOSSOS server in one step. This combines the functionality of [`convert`](convert.md#convert) and [`upload`](upload.md) into a single command, using a temporary directory for the intermediate dataset that is cleaned up automatically.

## Usage

```bash
webknossos remote-convert [OPTIONS] SOURCE
```

### Arguments

- **SOURCE**
    Path to your image data.
    Example: `/path/to/source/images`.

### Options

#### WEBKNOSSOS Server

- `--webknossos-url`
    URL to the WEBKNOSSOS instance.
    Can also be provided via the `WK_URL` environment variable.

- `--token`
    Authentication token for the WEBKNOSSOS instance (see https://webknossos.org/auth/token).
    Can also be provided via the `WK_TOKEN` environment variable.

- `--dataset-name`
    Name for the dataset on WEBKNOSSOS.
    If not provided, the source directory name is used.

- `--folder`
    WEBKNOSSOS dataset folder in which the dataset should be placed. Specify the folder path as a string, separated by `/`. Note that this is about the folders listed in the dataset dashboard, and is independent of the underlying storage location. Example: `Datasets/mySubfolder`.
    If not provided, the root folder is used.

#### Conversion

- `--voxel-size`
    The size of one voxel in the source data in nanometers.
    Example: `--voxel-size 11.0,11.0,20.0`.

- `--unit`
    The unit of the voxel size.
    Default: `nanometer`.

- `--layer-name`
    Name of the layer to be created.
    Default: `None`.

- `--category`
    The category of the layer that should be created.
    Options: `color`, `segmentation`.
    Default: `None` (auto-detected).

- `--data-format`
    Data format to store the target dataset.
    Options: `wkw`, `zarr`, `zarr3`.
    Default: `zarr3`.

- `--chunk-shape`
    Number of voxels to be stored as a chunk in the output format.
    Example: `--chunk-shape 32,32,32`.

- `--shard-shape`
    Number of voxels to be stored as a shard in the output format.
    Example: `--shard-shape 1024,1024,1024`.

- `--compress`
    Enable compression of the target dataset.
    Default: `True`.

- `--batch-size`
    Number of images to be processed in one batch (influences RAM consumption).
    When creating a WKW dataset, batch-size must be a multiple of chunk-shape's z dimension.
    When converting to Zarr or Zarr3, batch-size must be a multiple of the z dimension of the shard shape (chunk-shape x chunks-per-shard).
    Default: None (= z dimension of shard-shape).

#### Downsampling

- `--downsample`
    Downsample the dataset locally before uploading.
    Default: `True`.

- `--max-mag`
    Create downsampled magnifications up to the magnification specified by this argument.
    If omitted, the coarsest magnification will be determined by using the bounding box of the layer.
    Example: `--max-mag 2-2-1`.

- `--interpolation-mode`
    The interpolation mode that should be used.
    Options: `median`, `mode`, `nearest`, `bilinear`, `bicubic`.
    Default: `default` (= `mode` for segmentation, `median` for color).

- `--sampling-mode`
    The sampling mode to use.
    Options: `anisotropic`, `isotropic`, `constant_z`.
    Default: `anisotropic`.

#### Executor

- `--jobs`
    Number of processes to be spawned for parallel execution.
    Default: Number of CPU cores.

- `--distribution-strategy`
    Strategy to distribute the task across CPUs or nodes.
    Options: `multiprocessing`, `slurm`, `kubernetes`, `sequential`.
    Default: `multiprocessing`.

- `--job-resources`
    Necessary when using slurm as distribution strategy. Should be a JSON string.
    Example: `--job-resources '{"mem": "10M"}'`.

## Example Commands

### Convert and upload with default settings:
```bash
webknossos remote-convert --voxel-size 11.0,11.0,20.0 /path/to/source/images
```
Converts the images and uploads the dataset to the WEBKNOSSOS server specified by `WK_URL` or to the default url [https://webknossos.org](https://webknossos.org).

### Convert and upload with custom name and folder:
```bash
webknossos remote-convert \
    --voxel-size 11.0,11.0,20.0 \
    --webknossos-url https://webknossos.example.com \
    --token YOUR_TOKEN \
    --dataset-name my_dataset \
    --folder "Datasets/Experiments" \
    /path/to/source/images
```

### Convert and upload without downsampling:
```bash
webknossos remote-convert --voxel-size 11.0,11.0,20.0 --no-downsample /path/to/source/images
```

## Notes

- This command creates a temporary local dataset during conversion which is automatically cleaned up after upload.
- Downsampling is performed locally before uploading, so all magnifications are available on the server immediately.
- Provide authentication details via the `--token` option or the `WK_TOKEN` environment variable.
- Use the `--jobs` option to optimize performance for large datasets.
