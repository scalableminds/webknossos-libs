set -xe
cp -r testoutput/tiff testoutput/tiff_compress2
python -m wkcuber.compress \
  --jobs 2 \
  --layer_name color \
  testoutput/tiff_compress2
