#!/usr/bin/env bash
set -eEuo pipefail

TEST_WKW_PATH=testdata/WT1_wkw
TEST_WKW_ARCHIVE=testdata/WT1_wkw.tar.gz
TEST_TIFF_PATH=testdata/tiff_mag_2_reference
TEST_TIFF=testdata/tiff_mag_2_reference.tar.gz

OUTPUT_PATH=testoutput
SCRIPTS_BASEDIR=./tests/scripts

# Cleanup
if [ -d $OUTPUT_PATH ]; then
  echo "Removing $OUTPUT_PATH"
  rm -Rf $OUTPUT_PATH
fi

# Setup
if [ ! -d $TEST_WKW_PATH ]; then
  echo "Extracting $TEST_WKW_ARCHIVE into $TEST_WKW_PATH"
  tar -xzvf $TEST_WKW_ARCHIVE
fi

if [ ! -d $TEST_TIFF_PATH ]; then
  echo "Extracting $TEST_TIFF into $TEST_TIFF_PATH"
  mkdir -p $TEST_TIFF_PATH
  tar -xzvf $TEST_TIFF -C $TEST_TIFF_PATH
fi

# Note that pytest should be executed via `python -m`, since
# this will ensure that the current directory is added to sys.path
# (which is standard python behavior). This is necessary so that the imports
# refer to the checked out (and potentially modified) code.
poetry run python -m pytest -vv tests

poetry run sh ${SCRIPTS_BASEDIR}/tiff_cubing.sh
poetry run sh ${SCRIPTS_BASEDIR}/meta_generation.sh
poetry run sh ${SCRIPTS_BASEDIR}/downsampling.sh
poetry run sh ${SCRIPTS_BASEDIR}/upsampling.sh
rm -r $OUTPUT_PATH/tiff_upsampling
rm -r testdata/tiff_mag_2_reference

poetry run sh ${SCRIPTS_BASEDIR}/anisotropic_downsampling.sh
poetry run sh ${SCRIPTS_BASEDIR}/compression_and_verification.sh
rm -r $OUTPUT_PATH/tiff_compress
rm -r $OUTPUT_PATH/tiff_compress_broken

poetry run sh ${SCRIPTS_BASEDIR}/in_place_compression.sh
rm -r $OUTPUT_PATH/tiff_compress2
rm -r $OUTPUT_PATH/tiff_compress2.bak
rm -r $OUTPUT_PATH/tiff

poetry run sh ${SCRIPTS_BASEDIR}/tile_cubing.sh
rm -r $OUTPUT_PATH/temca2

poetry run sh ${SCRIPTS_BASEDIR}/simple_tiff_cubing.sh
rm -r $OUTPUT_PATH/tiff2

poetry run sh ${SCRIPTS_BASEDIR}/simple_tiff_cubing_no_compression.sh
rm -r $OUTPUT_PATH/tiff3

poetry run sh ${SCRIPTS_BASEDIR}/tiff_formats_cubing.sh
rm -r $OUTPUT_PATH/tiff4

poetry run sh ${SCRIPTS_BASEDIR}/knossos_conversion.sh
rm -r $OUTPUT_PATH/knossos

poetry run sh ${SCRIPTS_BASEDIR}/convert_raw.sh
rm -r $OUTPUT_PATH/raw

rm -r $OUTPUT_PATH
