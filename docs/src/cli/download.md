# `download`

The `download` command downloads a WEBKNOSSOS dataset or annotation from a provided URL. It supports specifying additional options such as authentication, partial download using a bounding box, specific layers, and magnifications.

## Usage

```bash
webknossos download [OPTIONS] TARGET
```

### Arguments

- **TARGET**  
    Path where the downloaded dataset or annotation will be saved.  
    Example: `/path/to/save/dataset`.

### Options

- `--url`  
    URL of your WEBKNOSSOS dataset or annotation.  
    Example: `--url https://webknossos.org/dataset/12345`.

- `--token`  
    Authentication token for your WEBKNOSSOS instance (e.g., https://webknossos.org/auth/token).  
    This option can be provided via the environment variable `WK_TOKEN`.

- `--bbox`  
    Define a bounding box for a partial download.  
    The input format is `x,y,z,width,height,depth`.  
    Example: `--bbox 0,0,0,10,10,10`.

- `--layer`  
    Specify one or more layers to download.  
    For multiple layers, use the option multiple times.  
    Example: `--layer color --layer segmentation`.

- `--mag`  
    Specify one or more magnifications for download.  
    The value can be a number or a minus-separated string (e.g. `2` or `2-2-2`).  
    For multiple mags, use the option multiple times.  
    Example: `--mag 1 --mag 2`.

- `--data-format`  
    Specify the data format of the downloaded dataset.  
    The value can be `zarr3`, `zarr` or `wkw`.  
    Example: `--data-format zarr3`.
    Default: `zarr3`.

## Example Commands

### Download a dataset:
```bash
webknossos download --url https://webknossos.org/dataset/12345 /path/to/save/dataset
```
This command downloads the entire dataset from the provided URL and saves it locally.

### Download a dataset with authentication and a bounding box:
```bash
WK_TOKEN=your-token webknossos download \
    --url https://webknossos.org/dataset/12345 \
    --bbox 0,0,0,10,10,10 \
    /path/to/save/dataset
```
This command downloads a partial dataset specified by a bounding box using an authentication token.

### Download an annotation:
```bash
webknossos download --url https://webknossos.org/annotation/67890 /path/to/save/annotation
```
This command downloads an annotation from the provided URL and saves it to the target path.

## Notes

- Ensure that the URL provided points to either a dataset or an annotation.
- If the dataset is not public, you must either set the `WK_TOKEN` environment variable or use the `--token` option.
- For partial downloads, specify the bounding box along with desired layers or magnifications.
- The command handles URL resolution internally, ensuring compatibility with short links.
- Errors are raised if the provided URL is invalid or does not point to a downloadable resource.