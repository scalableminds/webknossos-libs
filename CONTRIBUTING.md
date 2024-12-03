# Contributing Guide

**Welcome to the WEBKNOSSOS-libs contributing guide :sparkles:**

Thank you for taking the time to contribute to this project! The following is a set of guidelines for contributing to the different WEBKNOSSOS related Python libraries, which are part of the [WEBKNOSSOS-libs repository on GitHub](https://github.com/scalableminds/webknossos-libs). These are mostly guidelines, not rules. Use your best judgment, and feel free to propose changes to this document in a pull request.

## Code of Conduct

WEBKNOSSOS-libs and everyone contributing and collaborating on this project is expected to follow the [WEBKNOSSOS-libs Code of Conduct](CODE_OF_CONDUCT.md). Please report unacceptable behavior to [hello@webknossos.org](mailto:hello@webknossos.org).


## How can I help?

We welcome community feedback and contributions! We are happy to have

* [general feedback, observations and questions](#feedback-observations-and-questions) on the [image.sc forum](https://forum.image.sc/tag/webknossos),
* [feature suggestions and bug reports](#issues-feature-suggestions-and-bug-reports) as [issues on GitHub](https://github.com/scalableminds/webknossos-libs/issues/new),
* [documentation, examples and code contributions](#pull-requests-docs-and-code-contributions) as [pull requests on GitHub](https://github.com/scalableminds/webknossos-libs/compare).


## Feedback, Observations and Questions

We'd love to hear your feedback on the WEBKNOSSOS Python libraries!
We're also interested in hearing if these tools don't work for your usecase,
or if you have questions regarding their usage.

Please leave a message on the [image.sc forum](https://forum.image.sc/tag/webknossos)
using the `webknossos` tag to enable public communication on those topics.
If you prefer to share information only with the WEBKNOSSOS team, please write an email
to [hello@webknossos.org](mailto:hello@webknossos.org). For commercial support please
reach out to [scalable minds](https://scalableminds.com).


## Issues: Feature Suggestions and Bug Reports

We track feature requests and bug reports in the [WEBKNOSSOS-libs repository issues](https://github.com/scalableminds/webknossos-libs/issues).
Before opening a new issue, please do a quick search of existing issues to make sure your suggestion hasn’t already been added.
If your issue doesn’t already exist, and you’re ready to create a new one, make sure to state what you would like to implement, improve or bugfix.
Please use one of the provided templates to make this process easier for you.

You can submit an issue [here](https://github.com/scalableminds/webknossos-libs/issues/new)
(read more about [issues here](https://docs.github.com/en/issues)\).


### Report a Bug :lady_beetle:

When you find a bug, please double-check if an issue for the same bug exists already.
If that's not the case, please verify in the [documentation](https://docs.webknossos.org/api/webknossos.html)
that you use the API as intended. If that's the case, please
[add an issue using the bug report template](https://github.com/scalableminds/webknossos/issues/new?template=bug_report.md).


### Suggest a New Feature

If you are missing a feature to support your use-case, please consider the following points:

1. Please verify if this feature is directly related to WEBKNOSSOS.
   Does it belong into the WEBKNOSSOS Python libraries?
2. Double-check if an issue for this feature exists already. If there is one with a very similar scope,
   please considering commenting there.
3. If possible, consider how the implementation might look like (e.g. how would the public API change),
   as well as how this could be tested and presented in the examples.

Then, please [add an issue using the feature suggestion template](https://github.com/scalableminds/webknossos/issues/new?template=feature_suggestion.md).


## Pull Requests: Docs and Code Contributions

This project welcomes contributions and suggestions. Most contributions require you to
agree to a Contributor License Agreement (CLA) declaring that you have the right to,
and actually do, grant us the rights to use your contribution. For details, visit
https://cla-assistant.io/scalableminds/webknossos-libs.

When you submit a pull request, a CLA-bot will automatically determine whether you need
to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the
instructions provided by the bot. You will only need to do this once using our CLA.

If you want to fix a minor problem in the documentation, examples or other code, feel free to
[open a pull requests for it on GitHub](https://github.com/scalableminds/webknossos-libs/compare)
(read more about [pull requests here](https://docs.github.com/en/pull-requests).

For larger features and changes, we prefer that you open a new issue before creating a pull request.
Please include the following in your pull request:

* The pull request description should explain what you've done.
* Add tests for the new features.
* Comply with the coding styles.
* Adapt or add the documentation.
* Consider enhancing the examples.

The specific coding styles, test frameworks and documentation setup of WEBKNOSSOS-libs are described
in the following sections:


### Development

The [WEBKNOSSOS-libs repository](https://github.com/scalableminds/webknossos-libs) is structured as a mono-repo, containing multiple packages:

* `cluster_tools`
* `webknossos`
* (`docs`, see below for **Documentation**)

See below for specifics of the different packages. Let's have a look at the common tooling first:

* [**uv**](https://docs.astral.sh/uv) is used for dependency management and publishing.
  Use `uv sync --all-extras` in each package folder to install all dependencies for development.
  By default, this creates a [virtual environment](https://docs.python.org/3/tutorial/venv.html) for each package.
  To run commands inside this package, prefix them with `uv run`, e.g. `uv run python myscript.py`,
  or enter the virtual environment with `source .venv/bin/activate`.

  To install the dependencies for all sub-projects, run `make install`.
  
* **Tooling** we use across the sub-projects to enforce coding styles and tests:
    * `./format.sh`: ruff
    * `./lint.sh`: ruff
    * `./typecheck.sh`: mypy
    * `./test.sh`: pytest and custom scripts

  Those are also accessible via make commands from the top-level directory, running the respective scripts for each sub-project
  , e.g. `make format`, `make lint`, …

Internal workflows for scalable minds:

* **CI**: We use continuous integration with [github actions](https://github.com/scalableminds/webknossos-libs/actions),
  please see the `CI` workflow for details.
* **Releases** are triggered via the
  [`Automatic release` github action](https://github.com/scalableminds/webknossos-libs/actions/workflows/release.yml),
  using the `Run workflow` button in the topright corner.
  This updates the changelog and pushes a new tag, which triggers another CI run building and publishing the package.


#### `webknossos` package

The `webknossos` folder contains examples, which are not part of the package, but are tested via `tests/test_examples.py` and added to the documentation (see `docs/src/webknossos-py/examples`).

To run the `./test.sh` script it is necessary to install `proxay`. This is either done with [NPM](https://www.npmjs.com) or [yarn](https://yarnpkg.com/getting-started/install):
```bash
npm install --global proxay

# or if you're using yarn
yarn global add proxay
```

The tests also contain functionality for the WEBKNOSSOS client. There a two modes to run the tests:

1. `./test.sh --refresh-snapshots`, sending network requests to a WEBKNOSSOS instance:
  This expects a local WEBKNOSSOS setup with specific test data, which is shipped with WEBKNOSSOS. If you're starting and running WEBKNOSSOS manually, please use port 9000 (the default) and run the `tools/postgres/dbtool.js prepare-test-db` script in the WEBKNOSSOS repository (⚠️ this overwrites your local WEBKNOSSOS database). Alternatively, a `docker compose` setup is started automatically for the tests, see `./test.sh` and `tests/docker-compose.yml` for details. The network requests & response are recorded as "cassettes" by [proxay](https://github.com/airtasker/proxay), see next point:
2. `./test.sh` replays responses from previous network snapshots using [proxay](https://github.com/airtasker/proxay).

`./test.sh --store-durations` updates the durations for
[`pytest-split`](https://jerry-git.github.io/pytest-split),
which is used in the CI to split the tests for different runners.


#### `cluster_tools` package

For testing the `slurm` setup a `docker compose` setup is available. Please see the [respective Readme](https://github.com/scalableminds/webknossos-libs/blob/master/cluster_tools/README.md) for details.

For testing the `kubernetes` setup, we recommend a [Kubernetes-in-Docker setup](https://kind.sigs.k8s.io/).


### Documentation

We render a common documentation for WEBKNOSSOS itself and WEBKNOSSOS-libs from this repository using [mkdocs](https://www.mkdocs.org/). Source-files for the documentation are stored at `docs/src`:

* `docs/src/webknossos`: Server & Website documentation, linked from the [webknossos repository](https://github.com/scalableminds/webknossos) (must be available under `docs/wk-repo`, see below).
* `docs/src/api`: Generated using [pdoc](https://pdoc.dev) from Python docstrings.
* `docs/src/webknossos-py`: Documentation for the Python Package

The structure of the documentation page is given by `docs/mkdocs.yml`.

To generate the documentation locally, clone or link the WEBKNOSSOS repository to `docs/wk-repo` first and then start the documentation server
```shell
git clone --depth 1 git@github.com:scalableminds/webknossos.git docs/wk-repo
docs/generate.sh
```

You can use

* `docs/generate.sh` for hot-reloading markdown docs,
* `docs/generate.sh --api` to get the pure pdoc documentation, hot-reloading docstrings,
* `docs/generate.sh --persist` to persist the docs under docs/out.
