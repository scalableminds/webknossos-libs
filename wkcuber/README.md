# ⚠️⚠️⚠️ wkcuber ⚠️⚠️⚠️

⚠️⚠️⚠️ The `wkcuber`PyPI package is deprecated use `webknossos` instead. ⚠️⚠️⚠️

## How to continue working with wkcuber

- use `pip install webknossos` instead of `pip install wkcuber`
- replace `wkcuber` with `webknossos` in your requirements file (`requirements.txt`, `pyproject.toml`, `setup.py`, ...)
- if the `wkcuber` package is used by one of your dependencies it would be great if you would take the time to report the deprecation of the `wkcuber` package in the issue tracker


## Reasons for the deprecation

The `wkcuber` package on PyPI exists to prevent malicious actors from using the `wkcuber` package. The package `wkcuber` was used as standallone CLI to work with WEBKNOSSOS datasets, but was rebuild as a lightweight wrapper of the webknossos API. Therefore it was integratet into the `webknossos` package. `webknossos` is the actual package name now and should be used with python package managers, e.g. for:

- pip commands: `pip install webknossos`
- requirement files (`requirements.txt`, `pyproject.toml`, `setup.py`, ...)



## License
AGPLv3
Copyright scalable minds
