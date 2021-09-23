#!/usr/bin/env bash
set -eEuo pipefail

pylint -j4 webknossos
# pylint -j4 tests/**/*.py  # TODO add linting for tests
pylint -j4 examples/*.py
