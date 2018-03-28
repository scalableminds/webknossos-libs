# webKnossos cuber

[![CircleCI Status](https://circleci.com/gh/scalableminds/webknossos-cuber.svg?&style=shield)](https://circleci.com/gh/scalableminds/webknossos-cuber)

Easily create [WKW](https://github.com/scalableminds/webknossos-wrap) datasets for [webKnossos](https://webknossos.org).

The tools are modular components to allow easy integration into existing pipelines and workflows.

Created with [Python3](https://www.python.org/).

## Features

* `cubing`: Convert image stacks to WKW cubes (e.g., `tiff`, `jpg`, `png`)
* `convert`: Convert KNOSSOS cubes to WKW cubes
* `downsampling`: Create downsampled magnifications (with `median`, `mode` and linear interpolation modes)
* `compress`: Compress WKW cubes for efficient file storage (especially useful for segmentation data)
* `metadata`: Create metadata (with guessing of most parameters)
* Most modules support multiprocessing

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
Use the CI-built image: [scalableminds/webknossos-cuber](https://hub.docker.com/r/scalableminds/webknossos-cuber/). Example usage `docker run -v <host path>:/data --rm scalableminds/webknossos-cuber:wkw wkcuber.cubing  --layer_name color /data/source/color /data/target`.


## Usage

```
# Convert image files to wkw cubes
python -m wkcuber.cubing --layer_name color data/source/color data/target
python -m wkcuber.cubing --layer_name segmentation data/source/segmentation data/target

# Convert Knossos cubes to wkw cubes
python -m wkcuber.convert --layer_name color data/source data/target

# Create downsampled magnifications
python -m wkcuber.downsampling --layer_name color data/target
python -m wkcuber.downsampling --layer_name segmentation --interpolation_mode mode data/target

# Compress data in-place (mostly useful for segmentation)
python -m wkcuber.compress --layer_name segmentation data/target

# Compress data copy (mostly useful for segmentation)
python -m wkcuber.compress --layer_name segmentation data/target data/target_compress

# Create metadata
python -m wkcuber.metadata --name great_dataset --scale 11.24,11.24,25 data/target
```

# License
AGPLv3  
Copyright scalable minds
