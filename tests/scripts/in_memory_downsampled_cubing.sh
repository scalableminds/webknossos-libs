set -xe
mkdir -p testoutput/tiff
docker run \
  -v "${PWD}/testdata:/testdata" \
  -v "${PWD}/testoutput:/testoutput" \
  --rm \
  scalableminds/webknossos-cuber:${CIRCLE_BUILD_NUM} \
  wkcuber.cubing \
  --jobs 2 \
  --batch_size 8 \
  --layer_name color \
  --target_mag 2-2-1 \
  --interpolation_mode default \
  /testdata/tiff /testoutput/tiff4
[ -d testoutput/tiff4/color ]
[ -d testoutput/tiff4/color/2-2-1 ]
[ $(find testoutput/tiff4/color/2-2-1 -mindepth 3 -name "*.wkw" | wc -l) -eq 1 ]
