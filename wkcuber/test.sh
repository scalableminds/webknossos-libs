#!/usr/bin/env bash
set -eEuo pipefail

set +u
if [[ -z $RUNNER_OS || "$RUNNER_OS" == "Linux" ]]; then
    source ../_tooling/local_minio_setup.sh
fi
set -u

# Note that pytest should be executed via `python -m`, since
# this will ensure that the current directory is added to sys.path
# (which is standard python behavior). This is necessary so that the imports
# refer to the checked out (and potentially modified) code.
poetry run python -m pytest -vv tests
