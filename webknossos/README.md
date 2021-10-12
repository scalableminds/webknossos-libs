# webKnossos Python Library
[![PyPI version](https://img.shields.io/pypi/v/webknossos)](https://pypi.python.org/pypi/webknossos)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/webknossos.svg)](https://pypi.python.org/pypi/webknossos)
[![Build Status](https://img.shields.io/github/workflow/status/scalableminds/webknossos-libs/CI/master)](https://github.com/scalableminds/webknossos-libs/actions?query=workflow%3A%22CI%22)
[![Documentation](https://img.shields.io/badge/docs-passing-brightgreen.svg)](https://docs.webknossos.org/webknossos-py/index.html)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Python API for working with [webKnossos](https://webknossos.org) datasets, annotations, and for webKnossos server interaction.

## Features

- easy-to-use dataset API for reading/writing/editing raw 2D/3D image data and volume annotations/segmentation in webKnossos wrap (*.wkw) format
    - add/remove layers
    - update metadata ('datasource-properties.json`) 
    - up/downsample layers
    - compress layers 
    - add/remove magnifications
    - execute any of the `wkCuber` operations from your code
- manipulation of webKnossos skeleton annotations (*.nml) as Python objects
    - access to nodes, comments, trees, bounding boxes, metadata, etc.
    - create new skeleton annotation from Graph structures or Python objects
- interaction, connection & scripting with your webKnossos instance over the REST API
    - uploading/downloading annotations

## Installation
### Python 3 with pip from PyPi
- `webknossos` requires at least Python 3.7+

```
pip install wkcuber
```

## Examples
:todo:

## License
AGPLv3
Copyright scalable minds