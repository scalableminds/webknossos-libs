set -xe
mkdir -p testoutput/temca2
python -m wkcuber.tile_cubing \
  --jobs 2 \
  --batch_size 8 \
  --layer_name color \
  --scale 1,1,1 \
  testdata/temca2 testoutput/temca2
[ -d testoutput/temca2/color ]
[ -d testoutput/temca2/color/1 ]
[ $(find testoutput/temca2/color/1 -mindepth 3 -name "*.wkw" | wc -l) -eq 8 ]