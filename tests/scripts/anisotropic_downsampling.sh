set -xe
python -m wkcuber.downsampling \
  --jobs 2 \
  --from 1 \
  --max_mag 2 \
  --sampling_mode fix_z \
  --buffer_cube_size 128 \
  --layer_name color \
  testoutput/tiff

python -m wkcuber.downsampling \
  --jobs 2 \
  --from 2-2-1 \
  --max_mag 4 \
  --sampling_mode fix_z \
  --buffer_cube_size 128 \
  --layer_name color \
  testoutput/tiff
[ -d testoutput/tiff/color/2-2-1 ]
[ -d testoutput/tiff/color/4-4-1 ]
[ $(find testoutput/tiff/color/2-2-1 -mindepth 3 -name "*.wkw" | wc -l) -eq 1 ]
[ $(find testoutput/tiff/color/4-4-1 -mindepth 3 -name "*.wkw" | wc -l) -eq 1 ]

python -m wkcuber.downsampling \
  --jobs 2 \
  --from 4-4-1 \
  --max 16 \
  --buffer_cube_size 128 \
  --layer_name color \
  testoutput/tiff
[ -d testoutput/tiff/color/8-8-2 ]
[ -d testoutput/tiff/color/16-16-4 ]
[ $(find testoutput/tiff/color/8-8-2 -mindepth 3 -name "*.wkw" | wc -l) -eq 1 ]
[ $(find testoutput/tiff/color/16-16-4 -mindepth 3 -name "*.wkw" | wc -l) -eq 1 ]


