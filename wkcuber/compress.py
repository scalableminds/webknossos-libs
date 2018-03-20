import time
import wkw
import shutil
import logging
import numpy as np
from argparse import ArgumentParser
from os import path, makedirs

from .utils import \
    add_verbose_flag, add_jobs_flag, \
    open_wkw, WkwDatasetInfo, ParallelExecutor
from .metadata import detect_resolutions


def create_parser():
    parser = ArgumentParser()

    parser.add_argument(
        'source_path',
        help="Directory containing the source WKW dataset.")

    parser.add_argument(
        'target_path',
        help="Output directory for the compressed WKW dataset.",
        nargs='?',
        default=None)

    parser.add_argument(
        '--layer_name', '-l',
        help="Name of the cubed layer (color or segmentation)",
        default="color")

    parser.add_argument(
        '--mag', '-m',
        nargs='*',
        help="Magnification level",
        default=None)

    add_jobs_flag(parser)
    add_verbose_flag(parser)

    return parser


def compress_file_job(source_path, target_path):
    logging.info("Compressing '{}' to '{}'".format(source_path, target_path))
    ref_time = time.time()

    makedirs(path.dirname(target_path), exist_ok=True)
    wkw.File.compress(source_path, target_path)

    if not path.exists(target_path):
        raise Exception(
            "Did not create compressed file {}".format(target_path))

    logging.debug("Converting of '{}' took {:.8f}s".format(
        source_path, time.time() - ref_time))


if __name__ == '__main__':
    args = create_parser().parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    with_tmp_dir = args.target_path is None
    target_path = args.source_path + '.tmp' if with_tmp_dir else args.target_path

    if path.exists(target_path):
        logging.error("Target path '{}' already exists".format(target_path))
        exit(1)

    mags = args.mag
    if mags is None:
        mags = list(detect_resolutions(args.source_path, args.layer_name))
    mags.sort()

    for mag in mags:
        source_wkw_info = WkwDatasetInfo(
            args.source_path, args.layer_name, None, mag)
        target_mag_path = path.join(target_path, args.layer_name, str(mag))
        logging.info("Compressing mag {0} in '{1}'".format(
            mag, target_mag_path))

        with open_wkw(source_wkw_info) as source_wkw, \
                ParallelExecutor(args.jobs) as pool:
            source_wkw.compress(target_mag_path)
            for file in source_wkw.list_files():
                rel_file = path.relpath(file, source_wkw.root)
                pool.submit(compress_file_job, file,
                            path.join(target_mag_path, rel_file))

        logging.info("Mag {0} succesfully compressed".format(mag))

    if with_tmp_dir:
        makedirs(path.join(args.source_path + '.bak', args.layer_name))
        for mag in mags:
            shutil.move(
                path.join(args.source_path, args.layer_name, str(mag)),
                path.join(args.source_path + '.bak', args.layer_name, str(mag)))
            shutil.move(
                path.join(target_path, args.layer_name, str(mag)),
                path.join(args.source_path, args.layer_name, str(mag)))
        shutil.rmtree(target_path)
