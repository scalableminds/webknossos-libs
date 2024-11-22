# WEBKNOSSOS-libs
[![PyPI version](https://img.shields.io/pypi/v/webknossos)](https://pypi.python.org/pypi/webknossos)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/webknossos.svg)](https://pypi.python.org/pypi/webknossos)
[![Build Status](https://img.shields.io/github/actions/workflow/status/scalableminds/webknossos-libs/.github/workflows/ci.yml?branch=master)](https://github.com/scalableminds/webknossos-libs/actions?query=workflow%3A%22CI%22)
[![Documentation](https://img.shields.io/badge/docs-passing-brightgreen.svg)](https://docs.webknossos.org/webknossos-py/index.html)
[![Package Manager](https://img.shields.io/pypi/pyversions/uv.svg)](https://pypi.python.org/pypi/uv)
[![Code Style](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://docs.astral.sh/ruff/)

<img align="right" src="https://static.webknossos.org/logos/webknossos-icon-only.svg" alt="WEBKNOSSOS Logo" width="100" height="100"/>

## [WEBKNOSSOS Python API](webknossos)

### API

Python API for working with [WEBKNOSSOS](https://webknossos.org) datasets, annotations, and for WEBKNOSSOS server interactions.

Use this for:
- reading/writing/manipulating raw 2D/3D image data and volume annotations/segmentation in WEBKNOSSOS wrap (*.wkw) format
- handling/manipulation of WEBKNOSSOS datasets
- reading/writing/manipulating WEBKNOSSOS skeleton annotations (*.nml)
- up- & downloading annotations and datasets from your WEBKNOSSOS instance

[Read more in the docs.](https://docs.webknossos.org/webknossos-py/)

### CLI

CLI tool for creating and manipulating [WEBKNOSSOS](https://webknossos.org) [WKW](https://github.com/scalableminds/webknossos-wrap) datasets. WKW is a container format for efficiently storing large-scale 3D images as found in microscopy data.

Use this for:
- converting Tiff-stacks and other data formats for volume image data to WEBKNOSSOS-compatible *.wkw files from the CLI
- up/downsampling of *.wkw files to different magnification levels (image pyramid) from the CLI
- compressing your *.wkw files to save disk space from the CLI

[Read more in the docs.](https://docs.webknossos.org/webknossos-py/)

## [Cluster Tools](cluster_tools)
The `cluster_tools` package provides python `Executor` classes for distributing tasks on a slurm cluster or via multi processing.
