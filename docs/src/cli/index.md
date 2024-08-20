# WEBKNOSSOS CLI

Python CLI for creating and working with [WEBKNOSSOS](https://webknossos.org/) [WKW](https://github.com/scalableminds/webknossos-wrap) datasets. WKW is a container format for efficiently storing large, scale 3D image data as found in (electron) microscopy.

## Features

The WEBKNOSSOS CLI offers many useful commands to work with WEBKNOSSOS datasets:

- `webknossos compress`: Compress a WEBKNOSSOS dataset
- `webknossos convert`: Convert an image stack (e.g., `tiff`, `jpg`, `png`, `bmp`, `dm3`, `dm4`) to a WEBKNOSSOS dataset
- `webknossos convert-knossos`: Converts a KNOSSOS dataset to a WEBKNOSSOS dataset
- `webknossos convert-raw`: Converts a RAW image file to a WEBKNOSSOS dataset
- `webknossos convert-zarr`: Converts a Zarr dataset to a WEBKNOSSOS dataset 
- `webknossos download`: Download a dataset from a WEBKNOSSOS server as WKW format
- `webknossos downsample`: Downsample a WEBKNOSSOS dataset
- `webknossos merge-fallback`: Merge a volume layer of a WEBKNOSSOS dataset with an annotation
- `webknossos upload`: Upload a local WEBKNOSSOS dataset to a remote location
- `webknossos upsample`: Upsample a WEBKNOSSOS dataset

## Supported input formats

- Standard image formats, e.g. `tiff`, `jpg`, `png`, `bmp`
- Proprietary image formats, e.g. `dm3`
- Raw binary files

## Installation

### Python 3 with pip from PyPi

- `webknossos` requires at least Python 3.9

```bash
pip install "webknossos[all]"

# to install auto completion as well use:
webknossos --install-completion
```

## Usage

```bash
# Convert image stacks into wkw datasets
webknossos convert \
  --voxel-size 11.24,11.24,25 \
  --name great_dataset \
  data/source data/target


# Create downsampled magnifications
webknossos downsample data/target
webknossos downsample --layer-name color data/target

# Compress data in-place (mostly useful for segmentation)
webknossos compress data/target
webknossos compress data/target

# Convert Knossos cubes to wkw cubes
webknossos convert-knossos --layer-name color --voxel-size 11.24,11.24,25 data/source/mag1 data/target

# Convert RAW file to wkw file
webknossos convert-raw --layer-name color --voxel-size 10,10,30 --dtype uint8 --shape 2048,2048,1024 data/source/raw_file.raw data/target


```

### Parallelization

Most tasks can be configured to be executed in a parallelized manner. Via `--distribution-strategy` you can pass `multiprocessing`, `slurm` or `kubernetes`. The first can be further configured with `--jobs` and the latter via `--job-resources='{"mem": "10M"}'`. Use `--help` to get more information.


## Development

Make sure to install all the required dependencies using Poetry:
```bash
git clone git@github.com:scalableminds/webknossos-libs.git

cd webknossos-libs
pip install -r requirements.txt

cd webknossos
poetry install --all-extras
```

Please, format, lint, typecheck and unit test your code changes before merging them.
```bash
./format.sh
./lint.sh
./typecheck.sh
./test.sh
```

### Generate the API documentation
Run `docs/generate.sh` to open a server displaying the API docs. `docs/generate.sh --persist` persists the html to `docs/api`.

## License
AGPLv3
Copyright scalable minds
