# Contributions & Development

## How to contribute

We welcome community feedback and contributions! We are happy to have

* general feedback and questions on the [image.sc forum](https://forum.image.sc/tag/webknossos),
* feature requests and bug reports as [issues on GitHub](https://github.com/scalableminds/webknossos-libs/issues/new),
* documentation, examples and code contributions as [pull requests on GitHub](https://github.com/scalableminds/webknossos-libs/compare).


## Development

The [webknossos-libs repository](https://github.com/scalableminds/webknossos-libs) is structured as a mono-repo, containing multiple packages:

* `cluster_tools`
* `webknossos`
* `wkcuber`
* (`docs`, see below for **Documentation**)

See below for specifics of the different packages. Let's have a look at the common tooling first:

* [**poetry**](https://python-poetry.org) is used for dependency management and publishing.
  By default, it creates a [virtual environment](https://docs.python.org/3/tutorial/venv.html) for each package.
  To run commands inside this package, prefix them with `poetry run`, e.g. `poetry run python myscript.py`,
  or enter the virtual environment with `poetry shell`.
  The creation of a separate environment can be disabled (e.g. if you want to manage this manually),
  [see here for details](https://python-poetry.org/docs/configuration/#virtualenvscreate).
  To install the preferred version for this repository, run
  [`pip install -f requirements.txt`](https://github.com/scalableminds/webknossos-libs/blob/master/requirements.txt)
* **CI**: We use continuous integration with [github actions](https://github.com/scalableminds/webknossos-libs/actions),
  please see the `CI` workflow for details.
* **Tooling** we try to use across projects (still WIP, see [#581](https://github.com/scalableminds/webknossos-libs/issues/581)):
    * `format.sh`: black and isort
    * `lint.sh`: pylint
    * `typecheck.sh`: mypy
    * `test.sh`: pytest and custom scripts
* **Releases** are triggered via the
  [`Automatic release` github action](https://github.com/scalableminds/webknossos-libs/actions/workflows/release.yml),
  using the `Run workflow` button in the topright corner.
  This updates the changelog and pushes a new tag, which triggers another CI run building and publishing the package.


### webknossos package

The `webknossos` folder contains examples, which are not part of the package, but are tested via `tests/test_examples.py` and added to the documentation (see `docs/src/webknossos-py/examples`).

The tests also contain functionality for the webknossos client, sending network requests to a webknossos instance. For normal tests, those requests are read from previous network snapshots using [vcr.py](https://vcrpy.readthedocs.io) via [pytest-recording](https://github.com/kiwicom/pytest-recording).

This expects a local webknossos setup with specific test data, that is shipped with webknossos. If you're starting and running webknossos manually, please use port 9000 (the default) and run the `tools/postgres/prepareTestDb.sh` script in the webknossos repository (⚠️ this overwrites your local webknossos database). Alternatively, a docker-compose setup is started automatically for the tests, see `test.sh` and `tests/docker-compose.yml` for details.


### wkcuber package

Currently the test setup consists of different scripts as well as pytest tests. The following commands are run in CI:
```bash
tar -xzvf testdata/WT1_wkw.tar.gz
poetry run pytest tests
poetry run tests/scripts/all_tests.sh
```

There's also a `test.sh` which is outdated atm, see [issue #580](https://github.com/scalableminds/webknossos-libs/issues/580).


### cluster_tools package

To test the SLURM setup a docker-compose setup is availble. Please see the the [respective Readme](https://github.com/scalableminds/webknossos-libs/blob/master/cluster_tools/README.md) for details.


## Documentation

We render a common documentation for the webKnossos Server/Website and webknossos-libs from this repository using [mkdocs](https://www.mkdocs.org/). Source-files for the documentation are stored at `docs/src`:

* `docs/src/webknossos`: Server & Website documentation, linked from the [webknossos repository](https://github.com/scalableminds/webknossos) (must be available under `docs/wk-repo`, see below).
* `docs/src/api`: Generated using [pdoc](https://pdoc.dev) from Python docstrings.
* `docs/src/webknossos-py` & `docs/src/wkcuber` Documentation for the respective Python Packages

The structure of the documentation page is given by `docs/mkdocs.yml`.

To generate the documentation locally, use
```shell
# Clone or link the webknossos repo to docs/wk-repo:
git clone --depth 1 git@github.com:scalableminds/webknossos.git docs/wk-repo

# To generate the documentation, use one of
docs/generate.sh            # hot-reloading markdown docs
docs/generate.sh --api      # pure pdoc documentation, hot-reloading docstrings
docs/generate.sh --persist  # persists the docs under docs/out
```
