# Installing and Running the WEBKNOSSOS CLI

This document explains how to install the WEBKNOSSOS CLI tool either globally or within a Python environment using pip. It also covers how to run it using uv.

## Installation with pip

WEBKNOSSOS CLI requires Python 3.10 or higher.

### Global Installation or within a Virtual Environment

To install the tool, run:

```bash
pip install "webknossos[all]"
```

This command can be executed either in a global environment or within a virtual environment.

### Enabling Auto-Completion

After installation, enable shell auto-completion with:

```bash
webknossos --install-completion
```

## Running the CLI with uv

For running the WEBKNOSSOS CLI with [uv](https://docs.astral.sh/uv/getting-started/installation/), use the following command:

```bash
uv --with 'webknossos[all]' webknossos [COMMAND] [OPTIONS] [ARGUMENTS]
```

This ensures the latest webknossos CLI version is used to run the command.
