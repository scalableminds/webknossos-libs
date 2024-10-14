#!/usr/bin/env bash
set -eEuo pipefail

source local_wk_setup.sh


# Note that pytest should be executed via `python -m`, since
# this will ensure that the current directory is added to sys.path
# (which is standard python behavior). This is necessary so that the imports
# refer to the checked out (and potentially modified) code.
PYTEST="poetry run python -m pytest"


if [ $# -gt 0 ] && [ "$1" = "--refresh-snapshots" ]; then
    ensure_local_test_wk

    rm -rf tests/cassettes
    
    # Starts a proxy server in record mode on port 3000 and sets the HTTP_PROXY env var
    proxay --mode record --host http://localhost:9000 --tapes-dir tests/cassettes &

    shift
    $PYTEST "-m use_proxay $@"

    # Kill the proxy server
    kill %+

    stop_local_test_wk
else
    export_vars

    proxay --mode replay --tapes-dir tests/cassettes --rewrite-before-diff "s/[0-9]\{4\}-[0-9]\{2\}-[0-9]\{2\}_[0-9]\{2\}-[0-9]\{2\}-[0-9]\{2\}/2000-01-01_00-00-00/g" &
    $PYTEST "$@"
    kill %+
fi
