#!/usr/bin/env bash
set -eEuo pipefail

source local_wk_setup.sh

export_vars

export MINIO_ROOT_USER="TtnuieannGt2rGuie2t8Tt7urarg5nauedRndrur"
export MINIO_ROOT_PASSWORD="ANTN35UAENTS5UIAEATD"

# Minio is an S3 clone and is used as local test server
docker run \
  -p 8000:9000 \
  -e MINIO_ROOT_USER=$MINIO_ROOT_USER \
  -e MINIO_ROOT_PASSWORD=$MINIO_ROOT_PASSWORD \
  --name minio \
  --rm \
  -d \
  minio/minio server /data

stop_minio () {
    ARG=$?
    docker stop minio
    exit $ARG
}
trap stop_minio EXIT

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
