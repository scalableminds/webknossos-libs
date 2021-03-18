set -xe
mkdir -p testoutput/tiff2
python -m wkcuber.convert_image_stack_to_wkw \
  --jobs 2 \
  --batch_size 8 \
  --layer_name color \
  --max 8 \
  --scale "11.24,11.24,25"\
  --name awesome_data \
  testdata/tiff testoutput/tiff3
[ -d testoutput/tiff3/color ]
[ -d testoutput/tiff3/color/1 ]
[ $(find testoutput/tiff3/color/1 -mindepth 3 -name "*.wkw" | wc -l) -eq 1 ]
[ $(find testoutput/tiff3/color/2-1-2 -mindepth 3 -name "*.wkw" | wc -l) -eq 1 ]
[ $(find testoutput/tiff3/color/4-1-4 -mindepth 3 -name "*.wkw" | wc -l) -eq 1 ]
[ $(find testoutput/tiff3/color/8-2-8 -mindepth 3 -name "*.wkw" | wc -l) -eq 1 ]
[ -e testoutput/tiff3/datasource-properties.json ]