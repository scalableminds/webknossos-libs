set -xe
python -m wkcuber.compress \
  --jobs 2 \
  --layer_name color \
  testoutput/tiff testoutput/tiff_compress
[ -d testoutput/tiff_compress/color/1 ]
[ -d testoutput/tiff_compress/color/2 ]
[ -d testoutput/tiff_compress/color/4 ]
[ -d testoutput/tiff_compress/color/8 ]

echo "Generate metadata"
python -m wkcuber.metadata --name great_dataset --scale 11.24,11.24,25 testoutput/tiff
python -m wkcuber.metadata --name great_dataset --scale 11.24,11.24,25 testoutput/tiff_compress

echo "Check equality for uncompressed and compressed dataset"
python -m wkcuber.check_equality testoutput/tiff testoutput/tiff_compress

echo "Create broken copy of dataset"
cp -R testoutput/tiff_compress testoutput/tiff_compress_broken
rm -r testoutput/tiff_compress_broken/color/1/z0/y0/x0.wkw

echo "Compare original dataset to broken one and expect to determine difference"
if python -m wkcuber.check_equality testoutput/tiff testoutput/tiff_compress_broken ; then
    echo "Equality check did not fail even though the dataset is broken."
    exit 1
else
    echo "Equality check failed as expected for broken dataset."
    exit 0
fi