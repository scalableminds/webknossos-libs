import time
import wkw
import shutil
import numpy as np
from argparse import ArgumentParser
from os import path

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
        help="Magnification level",
        default=1)

    return parser


if __name__ == '__main__':
    args = create_parser().parse_args()

    with_tmp_dir = args.target_path is None
    target_path = args.source_path + '.tmp' if with_tmp_dir else args.target_path

    with open_wkw(args.source_path, args.layer_name, args.mag) as source_wkw:
        source_wkw.compress(path.join(target_path, args.layer_name, str(args.mag)), True)

    if with_tmp_dir:
        shutil.move(
            path.join(target_path, args.layer_name, str(args.mag)),
            path.join(args.source_path, args.layer_name, str(args.mag)))
