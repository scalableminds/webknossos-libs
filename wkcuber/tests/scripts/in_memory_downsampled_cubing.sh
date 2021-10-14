set -xe
mkdir -p testoutput/tiff
python -m wkcuber.cubing \
  --jobs 2 \
  --batch_size 8 \
  --layer_name color \
  --target_mag 2-2-1 \
  --interpolation_mode default \
  --scale 1 \
  testdata/tiff testoutput/in_memory_downsampled_tiff
[ -d testoutput/in_memory_downsampled_tiff/color ]
[ -d testoutput/in_memory_downsampled_tiff/color/2-2-1 ]
[ $(find testoutput/in_memory_downsampled_tiff/color/2-2-1 -mindepth 3 -name "*.wkw" | wc -l) -eq 1 ]
