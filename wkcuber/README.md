# WEBKNOSSOS cuber (wkcuber)
[![PyPI version](https://img.shields.io/pypi/v/wkcuber)](https://pypi.python.org/pypi/wkcuber)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/wkcuber.svg)](https://pypi.python.org/pypi/wkcuber)


Python CLI for creating and working with [WEBKNOSSOS](https://webknossos.org/) [WKW](https://github.com/scalableminds/webknossos-wrap) datasets. WKW is a container format for efficiently storing large, scale 3D image data as found in (electron) microscopy.

## Features

wkcuber offers some commands to work with WEBKNOSSOS datasets:

- `wkcuber compress`: Compress a WEBKNOSSOS dataset
- `wkcuber convert`: Convert an image stack (e.g., `tiff`, `jpg`, `png`, `bmp`, `dm3`, `dm4`) to a WEBKNOSSOS dataset
- `wkcuber download`: Download a dataset from a WEBKNOSSOS server as WKW format
- `wkcuber downsample`: Downsample a WEBKNOSSOS dataset
- `wkcuber upload`: Upload a local WEBKNOSSOS dataset to a remote location
- `wkcuber upsample`: Upsample a WEBKNOSSOS dataset

## Supported input formats

- Standard image formats, e.g. `tiff`, `jpg`, `png`, `bmp`
- Proprietary image formats, e.g. `dm3`
- Raw binary files

## Installation

### Python 3 with pip from PyPi

- `wkcuber` requires at least Python 3.8

```bash
pip install wkcuber

# to install auto completion as well use:
wkcuber --install-completion
```

## Usage

```bash
# Convert image stacks into wkw datasets
wkcuber convert \
  --voxel-size 11.24,11.24,25 \
  --name great_dataset \
  data/source data/target


# Create downsampled magnifications
wkcuber downsample data/target
wkcuber downsample --layer_name color data/target

# Compress data in-place (mostly useful for segmentation)
wkcuber compress --layer_name segmentation data/target
wkcuber compress data/target

# Old modules that are still usable until there is a typer replacement

# Convert Knossos cubes to wkw cubes
python -m wkcuber.convert_knossos --layer_name color data/source/mag1 data/target

# Convert NIFTI file to wkw file
python -m wkcuber.convert_nifti --layer_name color --scale 10,10,30 data/source/nifti_file data/target

# Convert folder with NIFTI files to wkw files
python -m wkcuber.convert_nifti --color_file one_nifti_file --segmentation_file --scale 10,10,30 another_nifti data/source/ data/target

# Convert RAW file to wkw file
python -m wkcuber.convert_raw --layer_name color --scale 10,10,30 --input_dtype uint8 --shape 2048,2048,1024 data/source/raw_file.raw data/target


```

### Parallelization

Most tasks can be configured to be executed in a parallelized manner. Via `--distribution_strategy` you can pass `multiprocessing`, `slurm` or `kubernetes`. The first can be further configured with `--jobs` and the latter via `--job_resources='{"mem": "10M"}'`. Use `--help` to get more information.


## Development

Make sure to install all the required dependencies using Poetry:
```bash
git clone git@github.com:scalableminds/webknossos-libs.git

cd webknossos-libs
pip install -r requirements.txt

cd wkcuber
poetry install --all-extras
```

Please, format, lint, and unit test your code changes before merging them.
```bash
poetry run black .
poetry run pylint -j4 wkcuber
poetry run pytest tests
```

Please, run the extended test suite:
```bash
tests/scripts/all_tests.sh
```

### Generate the API documentation
Run `docs/generate.sh` to open a server displaying the API docs. `docs/generate.sh --persist` persists the html to `docs/api`.

## License
AGPLv3
Copyright scalable minds