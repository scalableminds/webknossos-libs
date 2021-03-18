# webKnossos cuber (wkcuber)
[![PyPI version](https://img.shields.io/pypi/v/wkcuber)](https://pypi.python.org/pypi/wkcuber)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/wkcuber.svg)](https://pypi.python.org/pypi/wkcuber)
[![Build Status](https://img.shields.io/github/workflow/status/scalableminds/wkcuber/CI/master)](https://github.com/scalableminds/wkcuber/actions?query=workflow%3A%22CI%22)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Python library for creating and working with [webKnossos](https://webknossos.org) [WKW](https://github.com/scalableminds/webknossos-wrap) datasets. WKW is a container format for efficiently storing large, scale 3D image data as found in (electron) microscopy.

The tools are modular components to allow easy integration into existing pipelines and workflows.

## Features

* `wkcuber`: Convert supported input files to fully ready WKW datasets (includes type detection, downsampling, compressing and metadata generation)
* `wkcuber.convert_image_stack_to_wkw`: Convert image stacks to fully ready WKW datasets (includes downsampling, compressing and metadata generation)
* `wkcuber.export_wkw_as_tiff`: Convert WKW datasets to a tiff stack (writing as tiles to a `z/y/x.tiff` folder structure is also supported)
* `wkcuber.cubing`: Convert image stacks (e.g., `tiff`, `jpg`, `png`, `dm3`, `dm4`) to WKW cubes
* `wkcuber.tile_cubing`: Convert tiled image stacks (e.g. in `z/y/x.ext` folder structure) to WKW cubes
* `wkcuber.convert_knossos`: Convert KNOSSOS cubes to WKW cubes
* `wkcuber.convert_nifti`: Convert NIFTI files to WKW files (Currently without applying transformations).
* `wkcuber.downsampling`: Create downsampled magnifications (with `median`, `mode` and linear interpolation modes). Downsampling compresses the new magnifications by default (disable via `--no-compress`).
* `wkcuber.compress`: Compress WKW cubes for efficient file storage (especially useful for segmentation data)
* `wkcuber.metadata`: Create (or refresh) metadata (with guessing of most parameters)
* `wkcuber.recubing`: Read existing WKW cubes in and write them again specifying the WKW file length. Useful when dataset was written e.g. with file length 1.
* `wkcuber.check_equality`: Compare two WKW datasets to check whether they are equal (e.g., after compressing a dataset, this task can be useful to double-check that the compressed dataset contains the same data).
* Most modules support multiprocessing

## Supported input formats

* Standard image formats, e.g. `tiff`, `jpg`, `png`, `bmp`
* Proprietary image formats, e.g. `dm3`
* Tiled image stacks (used for Catmaid)
* KNOSSOS cubes
* NIFTI files

## Installation
### Python 3 with pip from PyPi
- `wkcuber` requires at least Python 3.6+

```
# Make sure to have lz4 installed:
# Mac: brew install lz4
# Ubuntu/Debian: apt-get install liblz4-1
# CentOS/RHEL: yum install lz4

pip install wkcuber
```

### Docker
Use the CI-built image: [scalableminds/webknossos-cuber](https://hub.docker.com/r/scalableminds/webknossos-cuber/). Example usage `docker run -v <host path>:/data --rm scalableminds/webknossos-cuber wkcuber --layer_name color --scale 11.24,11.24,25 --name great_dataset /data/source/color /data/target`.


## Usage

```
# Convert image stacks into wkw datasets
python -m wkcuber \
  --layer_name color \
  --scale 11.24,11.24,25 \
  --name great_dataset \
  data/source/color data/target

# Convert image files to wkw cubes
python -m wkcuber.cubing --layer_name color data/source/color data/target
python -m wkcuber.cubing --layer_name segmentation data/source/segmentation data/target

# Convert tiled image files to wkw cubes
python -m wkcuber.tile_cubing --layer_name color data/source data/target

# Convert Knossos cubes to wkw cubes
python -m wkcuber.convert_knossos --layer_name color data/source/mag1 data/target

# Convert NIFTI file to wkw file
python -m wkcuber.convert_nifti --layer_name color --scale 10,10,30 data/source/nifti_file data/target

# Convert folder with NIFTI files to wkw files
python -m wkcuber.convert_nifti --color_file one_nifti_file --segmentation_file --scale 10,10,30 another_nifti data/source/ data/target

# Create downsampled magnifications
python -m wkcuber.downsampling --layer_name color data/target
python -m wkcuber.downsampling --layer_name segmentation --interpolation_mode mode data/target

# Compress data in-place (mostly useful for segmentation)
python -m wkcuber.compress --layer_name segmentation data/target

# Compress data copy (mostly useful for segmentation)
python -m wkcuber.compress --layer_name segmentation data/target data/target_compress

# Create metadata
python -m wkcuber.metadata --name great_dataset --scale 11.24,11.24,25 data/target

# Refresh metadata so that new layers and/or magnifications are picked up
python -m wkcuber.metadata --refresh data/target

# Recubing an existing dataset
python -m wkcuber.recubing --layer_name color --dtype uint8 /data/source/wkw /data/target

# Check two datasets for equality
python -m wkcuber.check_equality /data/source /data/target
```

### Parallelization

Most tasks can be configured to be executed in a parallelized manner. Via `--distribution_strategy` you can pass `multiprocessing` or `slurm`. The first can be further configured with `--jobs` and the latter via `--job_resources='{"mem": "10M"}'`. Use `--help` to get more information.

## Development
Make sure to install all the required dependencies using Poetry:
```
pip install poetry
poetry install
```

Please, format, lint, and unit test your code changes before merging them.
```
poetry run black .
poetry run pylint -j4 wkcuber
poetry run pytest tests
```

Please, run the extended test suite:
```
tests/scripts/all_tests.sh
```

PyPi releases are automatically pushed when creating a new Git tag/Github release. 

## Test Data Credits
Excerpts for testing purposes have been sampled from:
- Dow Jacobo Hossain Siletti Hudspeth (2018). **Connectomics of the zebrafish's lateral-line neuromast reveals wiring and miswiring in a simple microcircuit.** eLife. [DOI:10.7554/eLife.33988](https://elifesciences.org/articles/33988)
- Zheng Lauritzen Perlman Robinson Nichols Milkie Torrens Price Fisher Sharifi Calle-Schuler Kmecova Ali Karsh Trautman Bogovic Hanslovsky Jefferis Kazhdan Khairy Saalfeld Fetter Bock (2018). **A Complete Electron Microscopy Volume of the Brain of Adult Drosophila melanogaster.** Cell. [DOI:10.1016/j.cell.2018.06.019](https://www.cell.com/cell/fulltext/S0092-8674(18)30787-6). License: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)

## License
AGPLv3
Copyright scalable minds
