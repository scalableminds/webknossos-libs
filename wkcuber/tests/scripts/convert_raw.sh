#!/usr/bin/env sh
set -xe
BASE_OUTPUT_DIR="testoutput/raw"
OUTPUT_DIR="$BASE_OUTPUT_DIR/dataset"
mkdir -p $OUTPUT_DIR

# Create dummy input data
for DTYPE in "uint8" "float32"; do
    echo "Test with dtype=$DTYPE"

    NAME="data_$DTYPE"
    INPUT_FILE="$BASE_OUTPUT_DIR/$NAME.raw"
    python -c "import numpy as np; np.arange(128**3, dtype=np.$DTYPE).reshape(128, 128, 128).tofile('$INPUT_FILE')"

    python -m wkcuber.convert_raw \
    --layer_name $NAME \
    --input_dtype $DTYPE \
    --shape 128,128,128 \
    --scale 11.24,11.24,25 \
    --no_compress \
    $INPUT_FILE $OUTPUT_DIR
    [ -d $OUTPUT_DIR/$NAME ]
    [ -d $OUTPUT_DIR/$NAME/1 ]
    [ $(find $OUTPUT_DIR/$NAME/1 -mindepth 3 -name "*.wkw" | wc -l) -eq 1 ]
    [ -e $OUTPUT_DIR/datasource-properties.json ]

done
