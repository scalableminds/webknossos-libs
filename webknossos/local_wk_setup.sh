function export_vars {
    export WK_TOKEN=1b88db86331a38c21a0b235794b9e459856490d70408bcffb767f64ade0f83d2bdb4c4e181b9a9a30cdece7cb7c65208cc43b6c1bb5987f5ece00d348b1a905502a266f8fc64f0371cd6559393d72e031d0c2d0cabad58cccf957bb258bc86f05b5dc3d4fff3d5e3d9c0389a6027d861a21e78e3222fb6c5b7944520ef21761e
    export WK_URL=http://localhost:9000
    export DOCKER_TAG=master__31064
}

function ensure_local_test_wk {
    export_vars
    WK_DOCKER_DIR="tests"

    if ! curl -sf localhost:9000/api/health; then
        echo "Using docker compose setup with the docker tag $DOCKER_TAG"
        echo "  To change this, please update DOCKER_TAG in local_wk_setup.sh"

        pushd $WK_DOCKER_DIR > /dev/null
        docker compose pull webknossos
        if [ ! -d binaryData/Organization_X/l4_sample ]; then
            mkdir -p binaryData/Organization_X
            pushd binaryData/Organization_X > /dev/null
            wget https://static.webknossos.org/data/l4_sample.zip
            unzip l4_sample.zip
            rm l4_sample.zip
            popd > /dev/null
        fi
        USER_UID=$(id -u) USER_GID=$(id -g) docker compose up -d --no-build webknossos
        stop_wk () {
            ARG=$?
            pushd $WK_DOCKER_DIR > /dev/null
            docker compose down
            popd > /dev/null
            exit $ARG
        }
        trap stop_wk EXIT
        while ! curl -sf localhost:9000/api/health; do
            sleep 5
        done
        # docker compose exec -T --user root webknossos sh -c "echo name,url,publicUrl,key,isScratch,isDeleted,allowsUpload,onlyAllowedOrganization,reportUsedStorageEnabled\n'http://localhost','http://localhost:9000','http://localhost:3000','something-secure',f,f,t,,f > test/db/dataStores.csv"
        # docker compose exec -T --user root webknossos sh -c "sed -i 's|publicUri = \${http.uri}|publicUri = \"localhost:3000\"|' conf/application.conf"
        docker compose exec -T --user root webknossos sh -c "sed -i \"s|f,t,'l4_sample'|t,t,'l4_sample'|\" test/db/dataSets.csv"
        # docker compose exec -T --user root webknossos sh -c "echo \"'59e9cfbdba632ac2ab8b23b7','localhost','Organization_X',,,'570b9f4e4bb848d0885ea917',,,,,,f,t,'l4_sample_dev','(11.239999771118164,11.239999771118164,28)','nanometer','',,,'2017-10-20T10:28:13.789Z','[]',{},'2017-10-20T10:28:13.789Z',f\" >> test/db/dataSets.csv"
        # docker compose exec -T --user root webknossos sh -c "cat test/db/dataSets.csv"
        OUT=$(docker compose exec -T webknossos tools/postgres/dbtool.js prepare-test-db 2>&1) || echo "$OUT"
        popd > /dev/null
    else
        echo "Using the already running local webknossos setup at localhost:9000"
    fi

    if ! curl -s -H "X-Auth-Token: $WK_TOKEN" localhost:9000/api/user | grep user_A@scalableminds.com > /dev/null; then
        echo "The login user user_A@scalableminds.com could not be found or changed."
        echo "Please ensure that the test-db is prepared by running this in the webknossos repo"
        echo "(⚠️ this overwrites your local webknossos database):"
        echo "tools/postgres/dbtool.js prepare-test-db"
        exit 1
    fi

    curl -s -X POST -H "X-Auth-Token: $WK_TOKEN" localhost:9000/data/triggers/checkInboxBlocking

    WK_ORG_VERSION="$(curl -s https://webknossos.org/api/buildinfo | tr ',"' "\n" | sed -n '/version/{n;n;p;q;}')"
    LOCAL_VERSION="$(curl -s http://localhost:9000/api/buildinfo | tr ',"' "\n" | sed -n '/version/{n;n;p;q;}')"

    if [ "$WK_ORG_VERSION" != "$LOCAL_VERSION" ]; then
        echo "The local webknossos version is $LOCAL_VERSION, differing from the webknossos.org version $WK_ORG_VERSION"
    fi
}


function stop_local_test_wk {
    if [ -n "${WK_DOCKER_DIR-}" ]; then
        pushd $WK_DOCKER_DIR > /dev/null
        docker compose down || true
        popd > /dev/null
    fi
}
