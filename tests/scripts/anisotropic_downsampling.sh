set -xe
docker run \
  -v "${PWD}/testoutput:/testoutput" \
  --rm \
  scalableminds/webknossos-cuber:${CIRCLE_BUILD_NUM} \
  wkcuber.downsampling \
  --jobs 2 \
  --from 1 \
  --anisotropic_target_mag 2-2-1 \
  --buffer_cube_size 128 \
  --layer_name color \
  /testoutput/tiff

docker run \
  -v "${PWD}/testoutput:/testoutput" \
  --rm \
  scalableminds/webknossos-cuber:${CIRCLE_BUILD_NUM} \
  wkcuber.downsampling \
  --jobs 2 \
  --from 2-2-1 \
  --anisotropic_target_mag 4-4-1 \
  --buffer_cube_size 128 \
  --layer_name color \
  /testoutput/tiff
[ -d testoutput/tiff/color/2-2-1 ]
[ -d testoutput/tiff/color/4-4-1 ]
[ $(find testoutput/tiff/color/2-2-1 -mindepth 3 -name "*.wkw" | wc -l) -eq 1 ]
[ $(find testoutput/tiff/color/4-4-1 -mindepth 3 -name "*.wkw" | wc -l) -eq 1 ]

docker run \
  -v "${PWD}/testoutput:/testoutput" \
  --rm \
  scalableminds/webknossos-cuber:${CIRCLE_BUILD_NUM} \
  wkcuber.downsampling \
  --jobs 2 \
  --from 4-4-1 \
  --max 16 \
  --buffer_cube_size 128 \
  --layer_name color \
  /testoutput/tiff
[ -d testoutput/tiff/color/8-8-2 ]
[ -d testoutput/tiff/color/16-16-4 ]
[ $(find testoutput/tiff/color/8-8-2 -mindepth 3 -name "*.wkw" | wc -l) -eq 1 ]
[ $(find testoutput/tiff/color/16-16-4 -mindepth 3 -name "*.wkw" | wc -l) -eq 1 ]


