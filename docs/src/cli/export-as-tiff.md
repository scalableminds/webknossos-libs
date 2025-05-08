# `export-as-tiff`

The `export-as-tiff` command exports a WEBKNOSSOS dataset to TIFF image data. It reads the dataset and saves it as tiff images, supporting options such as tiling, downsampling, and parallel execution.

## Usage

```bash
export-as-tiff [OPTIONS] SOURCE TARGET
```

### Arguments

- **SOURCE**  
    Path to your raw WEBKNOSSOS dataset.  
    Example: `/path/to/source/dataset`.

- **TARGET**  
    Target directory where the TIFF images will be saved.  
    Example: `/path/to/target/tiff_folder`.

### Options

- `--layer-name`  
    Name of the cubed layer (e.g., `color` or `segmentation`).  
    Default: `color`.

- `--bbox`  
    Bounding box to be exported. Format: `x,y,z,width,height,depth`.  
    If omitted, the full bounding box of the dataset is used.

- `--mag`  
    Maximum resolution for downsampling.  
    Provide a single number or a dash-separated string (e.g., `2` or `2-2-2`).  
    Default: `1`.

- `--name`  
    Base name for the TIFF files.  
    Each TIFF file will have this name followed by a slice index.

- `--downsample`  
    Downsample factor for each TIFF image.  
    Default: `1`.

- `--tiles-per-dimension`  
    Define tiling by specifying the number of tiles per dimension in the form `x,y`.  
    When set, each slice is exported as multiple tiled images.

- `--tile-size`  
    Specify the absolute size of each tile as `x,y`.  
    Use this if you prefer setting a fixed tile size instead of a number of tiles.

- `--batch-size`  
    Number of sections to buffer per job.  
    Default: Uses the Z-dimension of the default chunk shape.

- `--jobs`  
    Number of processes to spawn for parallel execution.  
    Default: Number of CPU cores.

- `--distribution-strategy`  
    Strategy to distribute the task across CPUs or nodes.  
    Options: `multiprocessing`, `slurm`, `kubernetes`, `sequential`. 
    Default: `multiprocessing`.

- `--job-resources`  
    Resources specification for jobs when using the SLURM strategy.  
    Example: `--job-resources='{"mem": "10M"}'`.

## Example Commands

### Export a dataset to TIFF images locally:
```bash
webknossos export-as-tiff /path/to/source/dataset /path/to/target/tiff_folder
```
This command exports all slices of the dataset to TIFF files in the target folder.

### Export with a specified layer and downsampling:
```bash
webknossos export-as-tiff --layer-name=color --downsample=2 /path/to/source/dataset /path/to/target/tiff_folder
```

### Export a specific bounding box with tiling:
```bash
webknossos export-as-tiff --bbox=0,0,0,512,512,100 --tiles-per-dimension=4,4 /path/to/source/dataset /path/to/target/tiff_folder
```

### Export using parallel processing with SLURM:
```bash
webknossos export-as-tiff --jobs=8 --distribution-strategy=slurm --job-resources='{"mem": "10M"}' /path/to/source/dataset /path/to/target/tiff_folder
```

## Notes

- Ensure the source dataset and target directory are accessible.
- The tiling options (`--tiles-per-dimension` and `--tile-size`) help manage very large datasets by splitting TIFF slices into multiple image tiles.
- Use the `--bbox` option to crop or pad the dataset as needed.
- Proper configuration of parallel execution options can significantly speed up the export process.
- For distributed processing, ensure the necessary resources and environment variables are properly set.
