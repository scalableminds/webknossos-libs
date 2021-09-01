set -xe

# create superfolder, so we can check how the autodetection deals with nested structures
mkdir -p testdata/superfolder/superfolder

# test wkw detection
python -m wkcuber.converter \
  --scale 11.24,11.24,25 \
  testdata/WT1_wkw testoutput/autodetection/wkw | grep -q "Already a WKW dataset."

# test wkw detection in subfolder
mv testdata/WT1_wkw testdata/superfolder/superfolder/WT1_wkw

python -m wkcuber.converter \
  --scale 11.24,11.24,25 \
  testdata/superfolder testoutput/autodetection/wkw | grep -q "Already a WKW dataset."

mv testdata/superfolder/superfolder/WT1_wkw testdata/WT1_wkw