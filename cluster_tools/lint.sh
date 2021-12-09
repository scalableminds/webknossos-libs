#!/usr/bin/env bash
set -eEuo pipefail

poetry run python -m pylint -j2 cluster_tools