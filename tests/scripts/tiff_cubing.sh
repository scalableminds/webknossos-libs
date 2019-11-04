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
  --wkw_file_len 32 \
  /testdata/tiff /testoutput/tiff
[ -d testoutput/tiff/color ]
[ -d testoutput/tiff/color/1 ]
[ $(find testoutput/tiff/color/1 -mindepth 3 -name "*.wkw" | wc -l) -eq 1 ]