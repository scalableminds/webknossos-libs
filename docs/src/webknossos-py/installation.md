# Installation

[![PyPI version](https://img.shields.io/pypi/v/webknossos)](https://pypi.python.org/pypi/webknossos)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/webknossos.svg)](https://pypi.python.org/pypi/webknossos)

The `webknossos` package requires at least Python 3.10.

You can install it from [pypi](https://pypi.org/project/webknossos/), e.g. via pip:

```bash
pip install webknossos
```

For extended file format conversation support it is necessary to install the optional dependencies:

```bash
pip install "webknossos[all]"
```

For working with Zeiss CZI microscopy data utilizing the`pylibczirw` package run:

```bash
pip install --extra-index-url https://pypi.scm.io/simple/ "webknossos[czi]"
```
