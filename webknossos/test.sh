#!/usr/bin/env bash
set -eEuo pipefail

source local_wk_setup.sh


# Note that pytest should be executed via `python -m`, since
# this will ensure that the current directory is added to sys.path
# (which is standard python behavior). This is necessary so that the imports
# refer to the checked out (and potentially modified) code.
PYTEST="uv run --all-extras --frozen python -m pytest --suppress-no-test-exit-code"

# Within the tests folder is a binaryData folder of the local running webknossos instance. This folder is cleaned up before running the tests.
find tests/binaryData/Organization_X -mindepth 1 -maxdepth 1 -type d ! -name 'l4_sample' ! -name 'e2006_knossos' -exec rm -rf {} +


if [ $# -gt 0 ] && [ "$1" = "--refresh-snapshots" ]; then
    ensure_local_test_wk

    rm -rf tests/cassettes
    
    # Starts a proxy server in record mode on port 3000 and sets the HTTP_PROXY env var
    proxay --mode record --host http://localhost:9000 --tapes-dir tests/cassettes &
    PROXAY_PID=$!

    shift
    $PYTEST "-m" "use_proxay" "$@"
    PYTEST_EXIT_CODE=$?

    # Kill the proxy server
    kill $PROXAY_PID
    wait $PROXAY_PID

    stop_local_test_wk

    exit $PYTEST_EXIT_CODE
elif [ $# -gt 0 ] && [ "$1" = "--debug-cassettes" ]; then
    # This will start a proxay server in replay mode so that the stored cassettes can be used for debugging tests.

    export_vars

    proxay --mode replay --tapes-dir tests/cassettes &
    PROXAY_PID=$!
    echo "Proxay server is running in replay mode. Press Ctrl+C to stop."
    trap 'kill $PROXAY_PID; exit' INT
    wait $PROXAY_PID
else
    export_vars

    proxay --mode replay --tapes-dir tests/cassettes 2>&1 > /dev/null &
    PROXAY_PID=$!

    $PYTEST "$@"
    PYTEST_EXIT_CODE=$?

    kill $PROXAY_PID

    exit $PYTEST_EXIT_CODE
fi
