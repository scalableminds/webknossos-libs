# WEBKNOSSOS CLI

Python CLI for creating and working with [WEBKNOSSOS](https://webknossos.org/) [WKW](https://docs.webknossos.org/webknossos/data/wkw.html) or [Zarr](https://docs.webknossos.org/webknossos/data/zarr.html) datasets. WKW is a container format for efficiently storing large, scale 3D image data as found in (electron) microscopy. Zarr is a community-driven, cloud-native data format for n-dimensional datasets.

## Features

The WEBKNOSSOS CLI offers many useful commands to work with WEBKNOSSOS datasets:

- [`webknossos compress`](compress.md): Compress a WEBKNOSSOS dataset
- [`webknossos convert`](convert.md#convert): Convert an image stack (e.g., `tiff`, `jpg`, `png`, `bmp`, `dm3`, `dm4`) to a WEBKNOSSOS dataset
- [`webknossos convert-knossos`](convert.md#convert-knossos): Converts a KNOSSOS dataset to a WEBKNOSSOS dataset
- [`webknossos convert-raw`](convert.md#convert-raw): Converts a RAW image file to a WEBKNOSSOS dataset
- [`webknossos convert-zarr`](convert.md#convert-zarr): Converts a Zarr dataset to a WEBKNOSSOS dataset 
- [`webknossos copy-dataset`](copy-dataset.md): Makes a copy of a WEBKNOSSOS dataset
- [`webknossos download`](download.md): Download a dataset from a WEBKNOSSOS server
- [`webknossos downsample`](downsample.md): Downsample a WEBKNOSSOS dataset
- [`webknossos merge-fallback`](merge-fallback.md): Merge a volume layer of a WEBKNOSSOS dataset with an annotation
- [`webknossos upload`](upload.md): Upload a local WEBKNOSSOS dataset to a remote location
- [`webknossos upsample`](upsample.md): Upsample a WEBKNOSSOS dataset
- [`webknossos export-as-tiff`](export-as-tiff.md): Export a part of a WEBKNOSSOS dataset as a TIFF sequence

### Environment Variables

To get access to services that require authentication, you can set the [environment variables](environment_variables.md):

- WK_TOKEN
- WK_URL
- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY
- S3_ENDPOINT_URL
- HTTP_BASIC_USER
- HTTP_BASIC_PASSWORD

### Parallelization

Most tasks can be configured to be executed in a parallelized manner. Via `--distribution-strategy` you can pass `multiprocessing`, `slurm` or `kubernetes`. The first can be further configured with `--jobs` and the latter via `--job-resources='{"mem": "10M"}'`. For a more detailed overview take a look at our [distribution strategies page](distribution_strategies.md).

## License
AGPLv3
Copyright scalable minds
