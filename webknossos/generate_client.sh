#!/usr/bin/env bash
set -eEuo pipefail

source local_wk_setup.sh

export_vars
ensure_local_test_wk

poetry run python __generate_client.py
