set -xe
mkdir -p testoutput/tiff2
docker run \
  -v "${PWD}/testdata:/testdata" \
  -v "${PWD}/testoutput:/testoutput" \
  --rm \
  scalableminds/webknossos-cuber:${CIRCLE_BUILD_NUM} \
  wkcuber \
  --jobs 2 \
  --batch_size 8 \
  --layer_name color \
  --max_mag 4 \
  --scale 11.24,11.24,25 \
  --name awesome_data \
  /testdata/tiff /testoutput/tiff2
[ -d testoutput/tiff2/color ]
[ -d testoutput/tiff2/color/1 ]
[ $(find testoutput/tiff2/color/1 -mindepth 3 -name "*.wkw" | wc -l) -eq 1 ]
[ -e testoutput/tiff2/datasource-properties.json ]