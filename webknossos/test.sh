#!/usr/bin/env bash
set -eEuo pipefail

source local_wk_setup.sh

export_vars

export MINIO_SECRET_KEY="TtnuieannGt2rGuie2t8Tt7urarg5nauedRndrur"
export MINIO_ACCESS_KEY="ANTN35UAENTS5UIAEATD"

# Minio is an S3 clone and is used as local test server
wget https://dl.min.io/server/minio/release/linux-amd64/minio -O ./minio
chmod +x ./minio
./minio server /tmp/minio_data &
MINIO_PID=$?

if [ $# -eq 1 ] && [ "$1" = "--refresh-snapshots" ]; then
    ensure_local_test_wk

    rm -rf tests/cassettes
    rm -rf tests/**/cassettes

    # Note that pytest should be executed via `python -m`, since
    # this will ensure that the current directory is added to sys.path
    # (which is standard python behavior). This is necessary so that the imports
    # refer to the checked out (and potentially modified) code.
    poetry run python -m pytest -vv --record-mode once -m "with_vcr"
else
    poetry run python -m pytest -vv --block-network -m "with_vcr"
fi
poetry run python -m pytest -vv --disable-recording -m "not with_vcr"

kill $MINIO_PID
