# Installing and Running the WEBKNOSSOS CLI

This document explains how to install the WEBKNOSSOS CLI tool. It also covers how to run it using uv.

## Installation with pip

WEBKNOSSOS CLI requires Python 3.10 or higher.

### Installation

To install the the webknossos CLI the webknossos package is required:

```bash
pip install "webknossos[all]"
```
[This guide](../webknossos-py/installation.md) explains other options to install this package.

### Enabling Auto-Completion

After installation, enable shell auto-completion with:

```bash
webknossos --install-completion
```

## Running the CLI with uv

For running the WEBKNOSSOS CLI with [uv](https://docs.astral.sh/uv), use the following command:

```bash
uv --with 'webknossos[all]' webknossos [COMMAND] [OPTIONS] [ARGUMENTS]
```

This ensures the latest webknossos CLI version is used to run the command.
