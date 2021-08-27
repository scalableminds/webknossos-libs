#!/usr/bin/env bash
set -eEuo pipefail

BASEDIR=./tests/scripts

sh ${BASEDIR}/tiff_cubing.sh
sh ${BASEDIR}/simple_tiff_cubing.sh
sh ${BASEDIR}/simple_tiff_cubing_no_compression.sh
sh ${BASEDIR}/tile_cubing.sh
sh ${BASEDIR}/meta_generation.sh
sh ${BASEDIR}/knossos_conversion.sh

mkdir -p testdata/tiff_mag_2_reference
tar -xzvf testdata/tiff_mag_2_reference.tar.gz -C testdata/tiff_mag_2_reference

sh ${BASEDIR}/anisotropic_downsampling.sh
sh ${BASEDIR}/compression_and_verification.sh
sh ${BASEDIR}/downsampling.sh
sh ${BASEDIR}/in_memory_downsampled_cubing.sh
sh ${BASEDIR}/in_place_compression.sh
sh ${BASEDIR}/simple_anisotropic_tiff_cubing.sh

rm -r testdata/tiff_mag_2_reference/