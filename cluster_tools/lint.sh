#!/usr/bin/env bash
set -eEuo pipefail

# Pylint doesn't lint files in directories that don't have an __init__.py
# This is not fixed by the --recursive=y flag (https://github.com/PyCQA/pylint/issues/352)
# Inspired by https://stackoverflow.com/questions/4210042/how-to-exclude-a-directory-in-find-command
find cluster_tools -iname "*.py" | xargs poetry run python -m pylint -j2
