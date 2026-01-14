# `upload-dataset`

The `upload-dataset` command uploads a local WEBKNOSSOS dataset to a specified WEBKNOSSOS server. It supports setting the target URL, authentication, renaming the dataset on upload, and parallel execution.

## Usage

```bash
webknossos upload [OPTIONS] SOURCE
```

### Arguments

- **SOURCE**  
    Path to your local WEBKNOSSOS dataset.  
    Example: `/path/to/local/dataset`.

### Options

- `--webknossos-url`  
    URL to the WEBKNOSSOS instance.  
    Can also be provided via the `WK_URL` environment variable. 

- `--token`  
    Authentication token for the WEBKNOSSOS instance (see https://webknossos.org/auth/token).  
    Can also be provided via the `WK_TOKEN` environment variable.

- `--dataset-name`  
    Alternative name to rename your dataset upon upload.  
    If not provided, the current dataset name is used.

- `--jobs`  
    Number of processes to spawn for parallel upload execution.  
    Default: 5.

- `--folder`  
    WEBKNOSSOS dataset folder in which the dataset should be placed. Specify the folder path as a string, separated by `/`. Note that this is about the folders listed in the dataset dashboard, and is independent of the underlying storage location. Example: `Datasets/mySubfolder`.  
    If not provided, the root folder is used.

### Environment Variables for Remote Paths

When using remote paths (e.g., HTTP or S3), configure the following environment variables:

#### HTTP Basic Authentication:
- `HTTP_BASIC_USER`  
- `HTTP_BASIC_PASSWORD`

#### S3 Configuration:
- `S3_ENDPOINT_URL`  
- `AWS_ACCESS_KEY_ID`  
- `AWS_SECRET_ACCESS_KEY`

## Example Commands

### Upload a dataset with default settings:
```bash
webknossos upload /path/to/local/dataset
```
Uploads a dataset to the WEBKNOSSOS server specified by `WK_URL` or to the default url [https://webknossos.org](https://webknossos.org).

### Upload a dataset with custom configuration:
```bash
webknossos upload --webknossos-url https://webknosos.example.com --token YOUR_TOKEN --dataset-name new_name --jobs 4 /path/to/local/dataset
```
Uploads the dataset with a new name and uses 4 parallel processes.

## Notes

- Ensure the dataset path is accessible and contains a valid WEBKNOSSOS dataset.
- Provide authentication details via the `--token` option or the `WK_TOKEN` environment variable.
- The command leverages parallel processing to speed up the upload process.
- Use `--dataset-name` if you wish to rename the dataset on the server.