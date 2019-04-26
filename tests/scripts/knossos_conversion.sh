set -xe
mkdir -p testoutput/knossos
docker run \
  -v "${PWD}/testdata:/testdata" \
  -v "${PWD}/testoutput:/testoutput" \
  --rm \
  scalableminds/webknossos-cuber:${CIRCLE_BUILD_NUM} \
  wkcuber.convert_knossos \
  --jobs 2 \
  --dtype uint8 \
  --layer_name color \
  --mag 1 \
  /testdata/knossos/color/1 /testoutput/knossos
[ -d testoutput/knossos/color ]
[ -d testoutput/knossos/color/1 ]
[ $(find testoutput/knossos/color/1 -mindepth 3 -name "*.wkw" | wc -l) -eq 1 ]