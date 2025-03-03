#!/usr/bin/env bash
set -eEuo pipefail

  cd tests
  PYTEST_EXECUTORS=multiprocessing,multiprocessing_with_pickling,sequential uv run --frozen python -m pytest -sv test_all.py test_multiprocessing.py

  echo "Tests for the kubernetes, dask and SLURM executors are only run in the CI"
