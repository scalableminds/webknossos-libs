#!/usr/bin/env bash
set -eEuo pipefail

source local_wk_setup.sh

export_vars


# Note that pytest should be executed via `python -m`, since
# this will ensure that the current directory is added to sys.path
# (which is standard python behavior). This is necessary so that the imports
# refer to the checked out (and potentially modified) code.
PYTEST="poetry run python -m pytest -s tests/test_tensorstore.py --suppress-no-test-exit-code"


if [ $# -gt 0 ] && [ "$1" = "--refresh-snapshots" ]; then
    ensure_local_test_wk

    rm -rf tests/cassettes
    rm -rf tests/**/cassettes

    shift
    $PYTEST --record-mode once -m "with_vcr" "$@"
    stop_local_test_wk
elif [ $# -gt 0 ] && [ "$1" = "--add-snapshots" ]; then
    ensure_local_test_wk
    shift
    $PYTEST --record-mode once -m "with_vcr" "$@"
    stop_local_test_wk
else
    $PYTEST --block-network -m "with_vcr" "$@"
fi
$PYTEST --disable-recording -m "not with_vcr" "$@"
