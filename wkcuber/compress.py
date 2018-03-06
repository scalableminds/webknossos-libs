import time
import wkw
import shutil
import numpy as np
from argparse import ArgumentParser
from os import path

from .metadata import detect_resolutions

def open_wkw(_path, layer_name, mag):
    return wkw.Dataset.open(path.join(_path, layer_name, str(mag)))

def create_parser():
    parser = ArgumentParser()

    parser.add_argument(
        'source_path',
        help="Directory containing the source WKW dataset.")

    parser.add_argument(
        'target_path',
        help="Output directory for the compressed WKW dataset.",
        default=None)

    parser.add_argument(
        '--layer_name', '-l',
        help="Name of the cubed layer (color or segmentation)",
        default="color")

    parser.add_argument(
        '--mag', '-m',
        nargs='*'
        help="Magnification level",
        default=None)

    parser.add_argument(
        '--verbose', '-v',
        help="Verbose output",
        dest="verbose",
        action='store_true')

    parser.set_defaults(verbose=False)

    return parser


if __name__ == '__main__':
    args = create_parser().parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    with_tmp_dir = args.target_path is None
    target_path = args.source_path + '.tmp' if with_tmp_dir else args.target_path

    mags = args.mag
    if mags is None:
        mags = detect_resolutions(args.source_path, args.layer_name)

    for mag in mags:
        target_mag_path = path.join(target_path, args.layer_name, str(mag))
        logging.info("Compressing mag {0} in {1}".format(mag, target_mag_path))

        with open_wkw(args.source_path, args.layer_name, mag) as source_wkw:
            source_wkw.compress(target_mag_path, True)

        if with_tmp_dir:
            shutil.move(
                path.join(target_path, args.layer_name, str(args.mag)),
                path.join(args.source_path, args.layer_name, str(args.mag)))

        logging.info("Mag {0} succesfully cubed".format(target_mag))
