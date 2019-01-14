set -xe
docker run \
  -v "${PWD}/testoutput:/testoutput" \
  --rm \
  scalableminds/webknossos-cuber:${CIRCLE_BUILD_NUM} \
  wkcuber.metadata \
  --name test_dataset \
  --scale 11.24,11.24,25 \
  /testoutput/tiff
[ -e testoutput/tiff/datasource-properties.json ]
jq --argfile a testdata/tiff/datasource-properties.fixture.json --argfile b testoutput/tiff/datasource-properties.json -n '$a == $b'
# diff testdata/tiff/datasource-properties.fixture.json testoutput/tiff/datasource-properties.json