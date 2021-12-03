#!/usr/bin/env bash
set -eEuo pipefail

if [ $# -eq 1 ] && [ "$1" = "--refresh-snapshots" ]; then
    if ! curl -sf localhost:9000/api/health; then
        WK_DOCKER_DIR="tests/.webknossos-server"
        if [ ! -d "$WK_DOCKER_DIR" ]; then
            git clone git@github.com:scalableminds/webknossos.git $WK_DOCKER_DIR --depth 1
            cp tests/wk-docker-compose.override.yml $WK_DOCKER_DIR/docker-compose.override.yml
        fi
        pushd $WK_DOCKER_DIR > /dev/null
        mkdir -p binaryData
        export DOCKER_TAG=21.11.0
        docker-compose pull webknossos
        USER_UID=$(id -u) USER_GID=$(id -g) docker-compose up -d --no-build webknossos
        popd > /dev/null
        stop_wk () {
            ARG=$?
            pushd $WK_DOCKER_DIR > /dev/null
            docker-compose down
            popd > /dev/null
            exit $ARG
        } 
        trap stop_wk EXIT
        while ! curl -sf localhost:9000/api/health; do
            sleep 5
        done
    fi
    rm -rf tests/cassettes
    poetry run pytest --record-mode once
else
    poetry run pytest --block-network
fi
