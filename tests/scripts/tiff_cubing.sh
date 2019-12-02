set -xe
mkdir -p testoutput/tiff
python -m wkcuber.cubing \
  --jobs 2 \
  --batch_size 8 \
  --layer_name color \
  --wkw_file_len 32 \
  testdata/tiff testoutput/tiff
[ -d testoutput/tiff/color ]
[ -d testoutput/tiff/color/1 ]
[ $(find testoutput/tiff/color/1 -mindepth 3 -name "*.wkw" | wc -l) -eq 1 ]