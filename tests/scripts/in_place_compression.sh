set -xe
cp -r testoutput/tiff testoutput/tiff_compress2
docker run \
  -v "${PWD}/testoutput:/testoutput" \
  --rm \
  scalableminds/webknossos-cuber:${CIRCLE_BUILD_NUM} \
  wkcuber.compress \
  --jobs 2 \
  --layer_name color \
  /testoutput/tiff_compress2
