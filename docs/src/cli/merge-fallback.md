# `merge-fallback`

The `merge-fallback` command merges a WEBKNOSSOS volume annotation layer with its fallback layer. It uses multi-processing or SLURM for parallel execution and accepts customization via options.

## Usage

```bash
merge-fallback [OPTIONS] TARGET SOURCE_ANNOTATION DATASET_DIRECTORY
```

### Arguments

- **TARGET**  
    Path to your WEBKNOSSOS output dataset.  
    Example: `/path/to/output/dataset` or `s3://bucket-name/output-dataset`.

- **SOURCE_ANNOTATION**  
    Path to the WEBKNOSSOS zip annotation file containing the fallback layer.  
    Example: `/path/to/source/annotation.zip`.

- **DATASET_DIRECTORY**  
    Path to your WEBKNOSSOS dataset folder.  
    Example: `/path/to/dataset/folder`.

### Options

- `--volume-layer-name`  
    Name of the volume layer to merge with the fallback layer.  
    Optional; if not provided, the default volume annotation layer is used.

- `--jobs`  
    Number of processes to be spawned for parallel execution.  
    Default: Number of CPU cores.

- `--distribution-strategy`  
    Strategy to distribute the task across CPUs or nodes.  
    Options: `multiprocessing`, `slurm`, `kubernetes`, `sequential`. 
    Default: `multiprocessing`.

- `--job-resources`  
    Specify resources for jobs when using the SLURM distribution strategy.  
    Example: `--job-resources='{"mem": "10M"}'`.

## Example Commands

### Merge fallback layer using default settings:
```bash
webknossos merge-fallback /path/to/output/dataset /path/to/source/annotation.zip /path/to/dataset/folder
```
Merges the fallback annotation into the volume layer using the default multiprocessing strategy.

### Merge fallback layer with a specific volume layer:
```bash
webknossos merge-fallback --volume-layer-name=my_volume /path/to/output/dataset /path/to/source/annotation.zip /path/to/dataset/folder
```

### Merge fallback layer with parallel execution and custom job settings:
```bash
webknossos merge-fallback --jobs=4 --distribution-strategy=slurm --job-resources='{"mem": "10M"}' /path/to/output/dataset /path/to/source/annotation.zip /path/to/dataset/folder
```
Uses 4 parallel processes with SLURM to merge the fallback layer, allocating custom memory to each job.

## Notes

- Ensure that the target dataset, source annotation, and dataset directory are correctly accessible.
- Set any necessary environment variables or credentials for remote paths.
- The SLURM distribution strategy requires specifying job resources via the `--job-resources` option.
- This command facilitates efficient merging of volume annotations with fallback layers by leveraging configurable parallel execution strategies.