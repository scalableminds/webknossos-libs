#!/usr/bin/env bash
set -eEuo pipefail

pylint -j4 webknossos
pylint -j4 examples/*.py
