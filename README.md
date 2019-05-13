# webKnossos cuber

[![CircleCI Status](https://circleci.com/gh/scalableminds/webknossos-cuber.svg?&style=shield)](https://circleci.com/gh/scalableminds/webknossos-cuber)

Easily create [WKW](https://github.com/scalableminds/webknossos-wrap) datasets for [webKnossos](https://webknossos.org).

The tools are modular components to allow easy integration into existing pipelines and workflows.

Created with [Python3](https://www.python.org/).

## Features

* `wkcuber`: Convert image stacks to fully ready WKW datasets (includes downsampling, compressing and metadata generation)
* `wkcuber.cubing`: Convert image stacks (e.g., `tiff`, `jpg`, `png`, `dm3`) to WKW cubes
* `wkcuber.tile_cubing`: Convert tiled image stacks (e.g. in `z/y/x.ext` folder structure) to WKW cubes
* `wkcuber.convert_knossos`: Convert KNOSSOS cubes to WKW cubes
* `wkcuber.downsampling`: Create downsampled magnifications (with `median`, `mode` and linear interpolation modes)
* `wkcuber.compress`: Compress WKW cubes for efficient file storage (especially useful for segmentation data)
* `wkcuber.metadata`: Create (or refresh) metadata (with guessing of most parameters)
* Most modules support multiprocessing

## Supported input formats

* Standard image formats, e.g. `tiff`, `jpg`, `png`, `bmp`
* Proprietary image formats, e.g. `dm3`
* Tiled image stacks (used for Catmaid)
* KNOSSOS cubes

## Installation
### Python3 with pip
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
```

### Parallelization

Most tasks can be configured to be executed in a parallelized manner. Via `--distribution_strategy` you can pass `multiprocessing` or `slurm`. The first can be further configured with `--jobs` and the latter via `--job_resources='{"mem": "10M"}'`. Use `--help` to get more information.

## Test data credits
Excerpts for testing purposes have been sampled from:
- Dow Jacobo Hossain Siletti Hudspeth (2018). **Connectomics of the zebrafish's lateral-line neuromast reveals wiring and miswiring in a simple microcircuit.** eLife. [DOI:10.7554/eLife.33988](https://elifesciences.org/articles/33988)
- Zheng Lauritzen Perlman Robinson Nichols Milkie Torrens Price Fisher Sharifi Calle-Schuler Kmecova Ali Karsh Trautman Bogovic Hanslovsky Jefferis Kazhdan Khairy Saalfeld Fetter Bock (2018). **A Complete Electron Microscopy Volume of the Brain of Adult Drosophila melanogaster.** Cell. [DOI:10.1016/j.cell.2018.06.019](https://www.cell.com/cell/fulltext/S0092-8674(18)30787-6). License: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)

## License
AGPLv3  
Copyright scalable minds
