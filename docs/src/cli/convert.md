# Convert Images

The conversion commands allow you to transform various types of images or image stacks into the WEBKNOSSOS dataset format. Each command is tailored for a specific input dataset type, ensuring efficient and accurate conversion. Below is an overview of the available commands and their usage.

## Available Commands

- [`convert`](#convert): Converts an image stack to a WEBKNOSSOS dataset.
- [`convert-knossos`](#convert-knossos): Converts a KNOSSOS dataset to a WEBKNOSSOS dataset.
- [`convert-raw`](#convert-raw): Converts a RAW dataset to a WEBKNOSSOS dataset.
- [`convert-zarr`](#convert-zarr): Converts a Zarr dataset to a WEBKNOSSOS dataset.

---

## `convert`

The `convert` command automatically detects an image stack and converts it into a WEBKNOSSOS dataset.

## Supported input formats

- Standard image formats, e.g. `tiff`, `jpg`, `png`, `bmp`
- Proprietary image formats, e.g. `dm3`
- Raw binary files

### Usage

```bash
convert [OPTIONS] SOURCE TARGET
```

### Arguments

- **SOURCE**  
    Path to the source image stack.  
    Example: `/path/to/source/images`.

- **TARGET**  
    Path to the target WEBKNOSSOS dataset.  
    Example: `/path/to/target/dataset`.

### Options

- `--voxel-size`  
    The size of one voxel in the source data in nanometers.  
    Example: `--voxel-size 11.0,11.0,20.0`.

- `--unit`  
    The unit of the voxel size.  
    Default: `nanometers`.

- `--layer-name`  
    Name of the layer to be created.  
    Default: `None`.

- `--data-format`  
    Data format to store the target dataset.  
    Options: `wkw`, `zarr`, `zarr3`  
    Default: `wkw`.

- `--chunk-shape`  
    Number of voxels to be stored as a chunk in the output format.  
    Example: `--chunk-shape 32,32,32`.

- `--shard-shape`  
    Number of voxels to be stored as a shard in the output format.  
    Example: `--shard-shape 1024,1024,1024`.

- `--compress`  
    Enable compression of the target dataset.  
    Default: `False`.

- `--jobs`  
    Number of processes to be spawned for parallel execution.  
    Default: Number of CPU cores.

- `--distribution-strategy`  
    Strategy to distribute the task across CPUs or nodes.  
    Options: `multiprocessing`, `slurm`, `kubernetes`, `sequential`. 
    Default: `multiprocessing`.

### Example Commands

Convert an image stack locally:

```bash
webknossos convert /path/to/source/images /path/to/target/dataset
```

Convert an image stack using parallel execution:

```bash
webknossos convert --jobs 4 /path/to/source/images /path/to/target/dataset
```

---

## `convert-knossos`

The `convert-knossos` command converts a KNOSSOS dataset into a WEBKNOSSOS dataset.

### Usage

```bash
convert-knossos [OPTIONS] SOURCE TARGET
```

### Arguments

- **SOURCE**  
    Path to the source KNOSSOS dataset.  
    Example: `/path/to/source/knossos`.

- **TARGET**  
    Path to the target WEBKNOSSOS dataset.  
    Example: `/path/to/target/dataset`.

### Options

- `--voxel-size`  
    The size of one voxel in the source data in nanometers.  
    Example: `--voxel-size 11.0,11.0,20.0`.

- `--unit`  
    The unit of the voxel size.  
    Default: `nanometers`.

- `--layer-name`  
    Name of the layer to be created.  
    Default: `color`.

- `--dtype`  
    Target datatype (e.g., `uint8`, `uint16`, `uint32`).  
    Default: `uint8`.

- `--data-format`  
    Data format to store the target dataset.  
    Options: `wkw`, `zarr`, `zarr3`  
    Default: `wkw`

- `--chunk-shape`  
    Number of voxels to be stored as a chunk in the output format.  
    Example: `--chunk-shape 32,32,32`.

- `--shard-shape`  
    Number of voxels to be stored as a shard in the output format.  
    Example: `--shard-shape 1024,1024,1024`.

- `--jobs`  
    Number of processes to be spawned for parallel execution.  
    Default: Number of CPU cores.

- `--distribution-strategy`  
    Strategy to distribute the task across CPUs or nodes.  
    Options: `multiprocessing`, `slurm`, `kubernetes`, `sequential`. 
    Default: `multiprocessing`.

### Example Commands

Convert a KNOSSOS dataset locally:

```bash
webknossos convert-knossos /path/to/source/knossos /path/to/target/dataset
```

Convert a KNOSSOS dataset with custom voxel size and parallel execution:

```bash
webknossos convert-knossos --voxel-size 11.0,11.0,20.0 --jobs 4 /path/to/source/knossos /path/to/target/dataset
```

---

## `convert-raw`

The `convert-raw` command converts a RAW dataset into a WEBKNOSSOS dataset.

### Usage

```bash
convert-raw [OPTIONS] SOURCE TARGET
```

### Arguments

- **SOURCE**  
    Path to the source RAW dataset.  
    Example: `/path/to/source/raw`.

- **TARGET**  
    Path to the target WEBKNOSSOS dataset.  
    Example: `/path/to/target/dataset`.

### Options

- `--shape`  
    Shape of the source dataset.  
    Example: `--shape 1024,1024,512`.

- `--order`  
    Input data storage layout: `C` (row-major) or `F` (column-major).  
    Default: `F`.

- `--voxel-size`  
    The size of one voxel in the source data in nanometers.  
    Example: `--voxel-size 11.0,11.0,20.0`.

- `--unit`  
    The unit of the voxel size.  
    Default: `nanometers`.

- `--layer-name`  
    Name of the layer to be created.  
    Default: `color`.

- `--dtype`  
    Target datatype (e.g., `uint8`, `uint16`, `uint32`).  
    Default: `uint8`.

- `--data-format`  
    Data format to store the target dataset. 
    Options: `wkw`, `zarr`, `zarr3`   
    Default: `wkw`.

- `--chunk-shape`  
    Number of voxels to be stored as a chunk in the output format.  
    Example: `--chunk-shape 32,32,32`.

- `--shard-shape`  
    Number of voxels to be stored as a shard in the output format.  
    Example: `--shard-shape 1024,1024,1024`.

- `--jobs`  
    Number of processes to be spawned for parallel execution.  
    Default: Number of CPU cores.

- `--distribution-strategy`  
    Strategy to distribute the task across CPUs or nodes.  
    Options: `multiprocessing`, `slurm`, `kubernetes`, `sequential`. 
    Default: `multiprocessing`.

### Example Commands

Convert a RAW dataset locally:

```bash
webknossos convert-raw --shape 1024,1024,512 /path/to/source/raw /path/to/target/dataset
```

Convert a RAW dataset using parallel execution:

```bash
webknossos convert-raw --shape 1024,1024,512 --jobs 4 /path/to/source/raw /path/to/target/dataset
```

---

## `convert-zarr`

The `convert-zarr` command converts a Zarr dataset into a WEBKNOSSOS dataset.

### Usage

```bash
convert-zarr [OPTIONS] SOURCE TARGET
```

### Arguments

- **SOURCE**  
    Path to the source Zarr dataset.  
    Example: `/path/to/source/zarr`.

- **TARGET**  
    Path to the target WEBKNOSSOS dataset.  
    Example: `/path/to/target/dataset`.

### Options

- `--voxel-size`  
    The size of one voxel in the source data in nanometers.  
    Example: `--voxel-size 11.0,11.0,20.0`.

- `--unit`  
    The unit of the voxel size.  
    Default: `nanometers`.

- `--layer-name`  
    Name of the layer to be created.  
    Default: `color`.

- `--data-format`  
    Data format to store the target dataset.
    Options: `wkw`, `zarr`, `zarr3`  
    Default: `wkw`.

- `--chunk-shape`  
    Number of voxels to be stored as a chunk in the output format.  
    Example: `--chunk-shape 32,32,32`.

- `--shard-shape`  
    Number of voxels to be stored as a shard in the output format.  
    Example: `--shard-shape 1024,1024,1024`.

- `--compress`  
    Enable compression of the target dataset.  
    Default: `True`.

- `--jobs`  
    Number of processes to be spawned for parallel execution.  
    Default: Number of CPU cores.

- `--distribution-strategy`  
    Strategy to distribute the task across CPUs or nodes.  
    Options: `multiprocessing`, `slurm`, `kubernetes`, `sequential`. 
    Default: `multiprocessing`.

### Example Commands

Convert a Zarr dataset locally:

```bash
webknossos convert-zarr /path/to/source/zarr /path/to/target/dataset
```

Convert a Zarr dataset using parallel execution:

```bash
webknossos convert-zarr --jobs 4 /path/to/source/zarr /path/to/target/dataset
```

Convert a Zarr dataset using SLURM with custom job resources:

```bash
webknossos convert-zarr --distribution-strategy slurm --job-resources '{"mem": "10M"}' /path/to/source/zarr /path/to/target/dataset
```

---

## Notes

- Ensure that the source and target paths are accessible and properly configured.
- Use the `--jobs` option to optimize performance for large datasets.
- For distributed execution, configure the `--distribution-strategy` and related options.
- Adjust options as needed to match your dataset specifications and desired output configuration.
