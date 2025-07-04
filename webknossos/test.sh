#!/usr/bin/env bash
set -eEuo pipefail

source local_wk_setup.sh

# Using forkserver instead of spawn is faster. Fork should never be used due to potential deadlock problems.
export MULTIPROCESSING_DEFAULT_START_METHOD=${MULTIPROCESSING_DEFAULT_START_METHOD:-forkserver}

# Note that pytest should be executed via `python -m`, since
# this will ensure that the current directory is added to sys.path
# (which is standard python behavior). This is necessary so that the imports
# refer to the checked out (and potentially modified) code.
PYTEST="uv run --all-extras --python ${PYTHON_VERSION:-3.13} -m pytest --suppress-no-test-exit-code -vv"

# Within the tests folder is a binaryData folder of the local running webknossos instance. This folder is cleaned up before running the tests.
# This find command gets all directories in binaryData/Organization_X except for the l4_sample and e2006_knossos directories and deletes them.
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
    trap 'kill $PROXAY_PID' EXIT

    stop_local_test_wk

    exit $PYTEST_EXIT_CODE
elif [ $# -gt 0 ] && [ "$1" = "--add-snapshots" ]; then
    ensure_local_test_wk

    # Starts a proxy server in record mode on port 3000 and sets the HTTP_PROXY env var
    proxay --mode record --host http://localhost:9000 --tapes-dir tests/cassettes &
    PROXAY_PID=$!

    shift
    $PYTEST "-m" "use_proxay" "$@"
    PYTEST_EXIT_CODE=$?
    trap 'kill $PROXAY_PID' EXIT

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
    trap 'kill $PROXAY_PID' EXIT

    $PYTEST "--timeout=360" "$@"
    PYTEST_EXIT_CODE=$?

    exit $PYTEST_EXIT_CODE
fi
