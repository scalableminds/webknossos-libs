set -xe
python -m wkcuber.compress \
  --jobs 2 \
  --layer_name color \
  testoutput/tiff testoutput/tiff_compress
[ -d testoutput/tiff_compress/color/1 ]
[ -d testoutput/tiff_compress/color/2 ]
[ -d testoutput/tiff_compress/color/4 ]
[ -d testoutput/tiff_compress/color/8 ]