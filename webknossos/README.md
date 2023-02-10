# WEBKNOSSOS Python Library
[![PyPI version](https://img.shields.io/pypi/v/webknossos)](https://pypi.python.org/pypi/webknossos)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/webknossos.svg)](https://pypi.python.org/pypi/webknossos)
[![Build Status](https://img.shields.io/github/actions/workflow/status/scalableminds/webknossos-libs/.github/workflows/ci.yml?branch=master)](https://github.com/scalableminds/webknossos-libs/actions?query=workflow%3A%22CI%22)
[![Documentation](https://img.shields.io/badge/docs-passing-brightgreen.svg)](https://docs.webknossos.org/webknossos-py)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Python API for working with [WEBKNOSSOS](https://webknossos.org) datasets, annotations, and for WEBKNOSSOS server interaction.

For the WEBKNOSSOS server, please refer to https://github.com/scalableminds/webknossos.

## Features

- easy-to-use dataset API for reading/writing/editing raw 2D/3D image data and volume annotations/segmentation in WEBKNOSSOS wrap (*.wkw) format
    - add/remove layers
    - update metadata (`datasource-properties.json`) 
    - up/downsample layers
    - compress layers 
    - add/remove magnifications
    - execute any of the `wkCuber` operations from your code
- manipulation of WEBKNOSSOS skeleton annotations (*.nml) as Python objects
    - access to nodes, comments, trees, bounding boxes, metadata, etc.
    - create new skeleton annotation from Graph structures or Python objects
- interaction, connection & scripting with your WEBKNOSSOS instance over the REST API
    - up- & downloading annotations and datasets

Please refer to [the documentation for further instructions](https://docs.webknossos.org/webknossos-py).

## Installation
The `webknossos` package requires at least Python 3.8.

You can install it from [pypi](https://pypi.org/project/webknossos/), e.g. via pip:

```bash
pip install webknossos
```

## Examples
See the [examples folder](examples) or the [the documentation](https://docs.webknossos.org/webknossos-py).

## Contributions & Development
Please see the [respective documentation page](https://docs.webknossos.org/webknossos-py/development.html).

## License
[AGPLv3](https://www.gnu.org/licenses/agpl-3.0.html)
Copyright [scalable minds](https://scalableminds.com)

## Test Data Credits
Excerpts for testing purposes have been sampled from:

* Dow Jacobo Hossain Siletti Hudspeth (2018). **Connectomics of the zebrafish's lateral-line neuromast reveals wiring and miswiring in a simple microcircuit.** eLife. [DOI:10.7554/eLife.33988](https://elifesciences.org/articles/33988)
* Zheng Lauritzen Perlman Robinson Nichols Milkie Torrens Price Fisher Sharifi Calle-Schuler Kmecova Ali Karsh Trautman Bogovic Hanslovsky Jefferis Kazhdan Khairy Saalfeld Fetter Bock (2018). **A Complete Electron Microscopy Volume of the Brain of Adult Drosophila melanogaster.** Cell. [DOI:10.1016/j.cell.2018.06.019](https://www.cell.com/cell/fulltext/S0092-8674(18)30787-6). License: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)
* Bosch Ackels Pacureanu et al (2022). **Functional and multiscale 3D structural investigation of brain tissue through correlative in vivo physiology, synchrotron microtomography and volume electron microscopy.** Nature Communications. [DOI:10.1038/s41467-022-30199-6](https://www.nature.com/articles/s41467-022-30199-6)
