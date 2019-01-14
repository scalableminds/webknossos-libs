set -xe
mkdir -p testoutput/tiff3
docker run \
  -v "${PWD}/testdata:/testdata" \
  -v "${PWD}/testoutput:/testoutput" \
  --rm \
  scalableminds/webknossos-cuber:${CIRCLE_BUILD_NUM} \
  wkcuber \
  --verbose \
  --jobs 2 \
  --batch_size 8 \
  --layer_name color \
  --max_mag 4 \
  --no_compress \
  --scale 11.24,11.24,25 \
  --name awesome_data \
  /testdata/tiff /testoutput/tiff3
[ -d testoutput/tiff3/color ]
[ -d testoutput/tiff3/color/1 ]
[ $(find testoutput/tiff3/color/1 -mindepth 3 -name "*.wkw" | wc -l) -eq 1 ]
[ -e testoutput/tiff3/datasource-properties.json ]