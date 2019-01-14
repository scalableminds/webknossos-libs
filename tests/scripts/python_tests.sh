set -xe
mkdir -p testoutput
docker run \
  -v "${PWD}/testdata:/testdata" \
  -v "${PWD}/testoutput:/testoutput" \
  --rm \
  --entrypoint "/bin/bash" \
  -w "/" \
  scalableminds/webknossos-cuber:${CIRCLE_BUILD_NUM} \
  -c "py.test /app/tests"