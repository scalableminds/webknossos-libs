set -xe
docker run \
  --rm \
  --entrypoint "/bin/bash" \
  -w "/" \
  scalableminds/webknossos-cuber:${CIRCLE_BUILD_NUM} \
  -c "black --check /app/wkcuber"