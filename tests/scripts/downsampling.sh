set -xe
docker run \
  -v "${PWD}/testoutput:/testoutput" \
  --rm \
  scalableminds/webknossos-cuber:${CIRCLE_BUILD_NUM} \
  wkcuber.downsampling \
  --jobs 2 \
  --max 8 \
  --buffer_cube_size 128 \
  --layer_name color \
  --isotropic \
  /testoutput/tiff
[ -d testoutput/tiff/color/2 ]
[ -d testoutput/tiff/color/4 ]
[ -d testoutput/tiff/color/8 ]
[ -n testoutput/tiff/color/16 ]
[ $(find testoutput/tiff/color/2 -mindepth 3 -name "*.wkw" | wc -l) -eq 1 ]
[ $(find testoutput/tiff/color/4 -mindepth 3 -name "*.wkw" | wc -l) -eq 1 ]
[ $(find testoutput/tiff/color/8 -mindepth 3 -name "*.wkw" | wc -l) -eq 1 ]
cmp --silent testoutput/tiff/color/2/z0/y0/x0.wkw testdata/tiff_mag_2_reference/color/2/z0/y0/x0.wkw
cmp --silent testoutput/tiff/color/2/header.wkw testdata/tiff_mag_2_reference/color/2/header.wkw