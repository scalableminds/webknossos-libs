set -xe
mkdir -p testoutput/tiff_pad
python -m wkcuber.cubing \
  --jobs 2 \
  --batch_size 8 \
  --layer_name color \
  --scale 1,1,1 \
  --start_z 4 \
  --pad \
  testdata/tiff_with_different_dimensions testoutput/tiff_pad
[ -d testoutput/tiff_pad/color ]
[ -d testoutput/tiff_pad/color/1 ]
[ -e testoutput/tiff_pad/datasource-properties.json ]