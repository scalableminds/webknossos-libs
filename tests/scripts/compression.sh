set -xe
docker run \
  -v "${PWD}/testoutput:/testoutput" \
  --rm \
  scalableminds/webknossos-cuber:${CIRCLE_BUILD_NUM} \
  wkcuber.compress \
  --silent \
  --jobs 2 \
  --layer_name color \
  /testoutput/tiff /testoutput/tiff_compress
[ -d testoutput/tiff_compress/color/1 ]
[ -d testoutput/tiff_compress/color/2 ]
[ -d testoutput/tiff_compress/color/4 ]
[ -d testoutput/tiff_compress/color/8 ]