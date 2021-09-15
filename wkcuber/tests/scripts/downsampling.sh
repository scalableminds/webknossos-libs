set -xe
python -m wkcuber.downsampling \
  --jobs 2 \
  --max 8 \
  --buffer_cube_size 128 \
  --layer_name color \
  --sampling_mode isotropic \
  testoutput/tiff
[ -d testoutput/tiff/color/2 ]
[ -d testoutput/tiff/color/4 ]
[ -d testoutput/tiff/color/8 ]
[ -n testoutput/tiff/color/16 ]
[ $(find testoutput/tiff/color/2 -mindepth 3 -name "*.wkw" | wc -l) -eq 1 ]
[ $(find testoutput/tiff/color/4 -mindepth 3 -name "*.wkw" | wc -l) -eq 1 ]
[ $(find testoutput/tiff/color/8 -mindepth 3 -name "*.wkw" | wc -l) -eq 1 ]
python tests/scripts/compare_wkw.py \
  testoutput/tiff/color/2 testdata/tiff_mag_2_reference/color/2
