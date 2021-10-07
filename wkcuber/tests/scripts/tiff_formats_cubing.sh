#!/usr/bin/env bash
set -xe
mkdir -p testoutput/tiff4

# simplest case: Only XY and I (alternative Z axis name) axis
python -m wkcuber.convert_image_stack_to_wkw \
  --jobs 2 \
  --batch_size 8 \
  --layer_name color \
  --max_mag 4 \
  --target_mag 2 \
  --scale 1,1,1 \
  --name awesome_data \
  --no_compress \
  --wkw_file_len 8 \
  testdata/various_tiff_formats/test_I.tif testoutput/tiff4
[ -d testoutput/tiff4/color ]
[ -d testoutput/tiff4/color/2 ]
[ $(find testoutput/tiff4/color/2 -mindepth 3 -name "*.wkw" | wc -l) -eq 1 ]
[ -e testoutput/tiff4/datasource-properties.json ]
rm -r testoutput/tiff4/*

# to make output wk_compatible, these files require multiple layer
python -m wkcuber.convert_image_stack_to_wkw \
  --jobs 2 \
  --batch_size 8 \
  --layer_name color \
  --max_mag 1 \
  --scale 1,1,1 \
  --name awesome_data \
  --no_compress \
  --wkw_file_len 8 \
  testdata/various_tiff_formats/test_C.tif testoutput/tiff4
[ -d testoutput/tiff4/color_0 ]
[ -d testoutput/tiff4/color_1 ]
[ -d testoutput/tiff4/color_2 ]
[ -d testoutput/tiff4/color_3 ]
[ -d testoutput/tiff4/color_4 ]
[ -d testoutput/tiff4/color_0/1 ]
[ $(find testoutput/tiff4/color_0/1 -mindepth 3 -name "*.wkw" | wc -l) -eq 1 ]
[ -e testoutput/tiff4/datasource-properties.json ]
rm -r testoutput/tiff4/*

python -m wkcuber.convert_image_stack_to_wkw \
  --jobs 2 \
  --batch_size 8 \
  --layer_name color \
  --max_mag 1 \
  --scale 1,1,1 \
  --name awesome_data \
  --no_compress \
  --wkw_file_len 8 \
  testdata/various_tiff_formats/test_S.tif testoutput/tiff4
[ -d testoutput/tiff4/color_0 ]
[ -d testoutput/tiff4/color_1 ]
[ -d testoutput/tiff4/color_2 ]
[ -d testoutput/tiff4/color_0/1 ]
[ $(find testoutput/tiff4/color_0/1 -mindepth 3 -name "*.wkw" | wc -l) -eq 1 ]
[ -e testoutput/tiff4/datasource-properties.json ]
rm -r testoutput/tiff4/*

python -m wkcuber.convert_image_stack_to_wkw \
  --jobs 2 \
  --batch_size 8 \
  --layer_name color \
  --max_mag 1 \
  --scale 1,1,1 \
  --name awesome_data \
  --no_compress \
  --wkw_file_len 8 \
  testdata/various_tiff_formats/test_CS.tif testoutput/tiff4
[ -d testoutput/tiff4/color_0 ]
[ -d testoutput/tiff4/color_1 ]
[ -d testoutput/tiff4/color_2 ]
[ -d testoutput/tiff4/color_3 ]
[ -d testoutput/tiff4/color_4 ]
[ -d testoutput/tiff4/color_0/1 ]
[ $(find testoutput/tiff4/color_0/1 -mindepth 3 -name "*.wkw" | wc -l) -eq 1 ]
[ -e testoutput/tiff4/datasource-properties.json ]
rm -r testoutput/tiff4/*

# test wk incompatible configs, program should fail because force parameter was not given
if python -m wkcuber.convert_image_stack_to_wkw \
    --jobs 2 \
    --batch_size 8 \
    --layer_name color \
    --max_mag 1 \
    --scale 1,1,1 \
    --prefer_channels \
    --name awesome_data \
    --no_compress \
    --wkw_file_len 8 \
    testdata/various_tiff_formats/test_C.tif testoutput/tiff4; then
  echo "Conversion worked although invalid config should have been detected."
  exit 1
else
  echo "Conversion did not work as expected due to invalid config."
fi

# power user configuration: should only create single layer
python -m wkcuber.convert_image_stack_to_wkw \
  --jobs 2 \
  --batch_size 8 \
  --layer_name color \
  --max_mag 1 \
  --scale 1,1,1 \
  --channel_index 3 \
  --sample_index 2 \
  --name awesome_data \
  --no_compress \
  --wkw_file_len 8 \
  testdata/various_tiff_formats/test_CS.tif testoutput/tiff4
[ -d testoutput/tiff4/color ]
[ -d testoutput/tiff4/color/1 ]
[ $(find testoutput/tiff4/color/1 -mindepth 3 -name "*.wkw" | wc -l) -eq 1 ]
[ -e testoutput/tiff4/datasource-properties.json ]
