#!/usr/bin/env bash
set -eEuo pipefail

if [ -d "./testoutput" ]; then rm -Rf ./testoutput; fi

BASEDIR=./tests/scripts

mkdir -p testdata/tiff_mag_2_reference
tar -xzvf testdata/tiff_mag_2_reference.tar.gz -C testdata/tiff_mag_2_reference

sh ${BASEDIR}/tiff_cubing.sh
sh ${BASEDIR}/meta_generation.sh
sh ${BASEDIR}/downsampling.sh
sh ${BASEDIR}/upsampling.sh
rm -r testoutput/tiff_upsampling
rm -r testdata/tiff_mag_2_reference

sh ${BASEDIR}/anisotropic_downsampling.sh
sh ${BASEDIR}/compression_and_verification.sh
rm -r testoutput/tiff_compress
rm -r testoutput/tiff_compress_broken

sh ${BASEDIR}/in_place_compression.sh
rm -r testoutput/tiff_compress2
rm -r testoutput/tiff_compress2.bak
rm -r testoutput/tiff

sh ${BASEDIR}/tile_cubing.sh
rm -r testoutput/temca2

sh ${BASEDIR}/simple_tiff_cubing.sh
rm -r testoutput/tiff2

sh ${BASEDIR}/simple_tiff_cubing_no_compression.sh
rm -r testoutput/tiff3

sh ${BASEDIR}/tiff_formats_cubing.sh
rm -r testoutput/tiff4

sh ${BASEDIR}/knossos_conversion.sh
rm -r testoutput/knossos

sh ${BASEDIR}/convert_raw.sh
rm -r testoutput/raw
