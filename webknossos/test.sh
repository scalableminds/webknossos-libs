#!/usr/bin/env bash
set -eEuo pipefail

source local_wk_setup.sh

# export_vars


# Note that pytest should be executed via `python -m`, since
# this will ensure that the current directory is added to sys.path
# (which is standard python behavior). This is necessary so that the imports
# refer to the checked out (and potentially modified) code.
PYTEST="poetry run python -m pytest"


if [ $# -gt 0 ] && [ "$1" = "--refresh-snapshots" ]; then
    ensure_local_test_wk

    rm -rf tests/cassettes
    
    proxay --mode record --tapes-dir tests/cassettes --redact-headers  &
    export http_proxy=http://localhost:3000
    shift
    $PYTEST "$@"
    stop_local_test_wk
else
    export_vars

    proxay --mode mimic --host http://localhost:9000 --tapes-dir tests/cassettes 2>&1 > /dev/null &
    export http_proxy=http://localhost:3000
    $PYTEST "$@"
fi
