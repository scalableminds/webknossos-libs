# Contributions & Development

## How to contribute

We welcome community feedback and contributions! We are happy to have

* general feedback and questions on the [image.sc forum](https://forum.image.sc/tag/webknossos),
* feature requests and bug reports as [issues on GitHub](https://github.com/scalableminds/webknossos-libs/issues/new),
* documentation, examples and code contributions as [pull requests on GitHub](https://github.com/scalableminds/webknossos-libs/compare),


## Development

### General
* poetry (venv, poetry run, …)
* lint, format, typecheck, test.sh
* CI
* releases (gh action)

### webknossos package
* tests: network snapshots
* examples (tested)

### wkcuber package
* test CLIs


## Generating the Documentation

We render a common documentation for the webKnossos Server/Website and webknossos-libs from this repository. Source-files for the documentation are stored at `docs/src`:

* `docs/src/webknossos`: Server & Website documentation, linked from the [webknossos repository](https://github.com/scalableminds/webknossos) (must be available under `docs/wk-repo`, see below).
* `docs/src/api`: Generated using [https://pdoc.dev] from Python docstrings.
* `docs/src/webknossos-py` & `docs/src/wkcuber` Documentation for the respective Python Packages

The structure of the documentation page is given by `docs/mkdocs.yml`.

To generate the documentation locally, use
```shell
git clone git@github.com:scalableminds/webknossos.git docs/wk-repo
docs/generate.sh            # hot-reloading markdown docs
docs/generate.sh --api      # pure pdoc documentation, hot-reloading docstrings
docs/generate.sh --persist  # persists the docs under docs/out
```
