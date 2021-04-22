from pathlib import Path

import shutil
import logging
from argparse import ArgumentParser, Namespace
from os import path, makedirs

from wkcuber.api.Dataset import WKDataset
from .mag import Mag

from .utils import (
    add_verbose_flag,
    add_distribution_flags,
    setup_logging,
)
from typing import List

BACKUP_EXT = ".bak"


def create_parser() -> ArgumentParser:
    parser = ArgumentParser()

    parser.add_argument(
        "source_path", help="Directory containing the source WKW dataset."
    )

    parser.add_argument(
        "target_path",
        help="Output directory for the compressed WKW dataset.",
        nargs="?",
        default=None,
    )

    parser.add_argument(
        "--layer_name",
        "-l",
        help="Name of the cubed layer (color or segmentation)",
        default="color",
    )

    parser.add_argument(
        "--mag", "-m", nargs="*", help="Magnification level", default=None
    )

    add_verbose_flag(parser)
    add_distribution_flags(parser)

    return parser


def compress_mag(
    source_path: str,
    layer_name: str,
    target_path: str,
    mag: Mag,
    args: Namespace = None,
) -> None:
    WKDataset(source_path).get_layer(layer_name).get_mag(mag).compress(
        target_path=Path(target_path), args=args
    )


def compress_mag_inplace(
    target_path: str, layer_name: str, mag: Mag, args: Namespace = None
) -> None:
    WKDataset(target_path).get_layer(layer_name).get_mag(mag).compress(args=args)


def compress_mags(
    source_path: str,
    layer_name: str,
    target_path: str = None,
    mags: List[Mag] = None,
    args: Namespace = None,
) -> None:
    if target_path is None:
        target = source_path + ".tmp"
    else:
        target = target_path

    layer = WKDataset(source_path).get_layer(layer_name)
    if mags is None:
        mags = [Mag(mag_name) for mag_name in layer.mags.keys()]

    for mag_name, mag_ds in WKDataset(source_path).get_layer(layer_name).mags.items():
        if Mag(mag_name) in mags:
            mag_ds.compress(target_path=Path(target), args=args)

    if target_path is None:
        makedirs(path.join(source_path + BACKUP_EXT, layer_name), exist_ok=True)
        for mag in mags:
            shutil.move(
                path.join(source_path, layer_name, str(mag)),
                path.join(source_path + BACKUP_EXT, layer_name, str(mag)),
            )
            shutil.move(
                path.join(target, layer_name, str(mag)),
                path.join(source_path, layer_name, str(mag)),
            )
        shutil.rmtree(target)
        logging.info(
            "Old files are still present in '{0}.bak'. Please remove them when not required anymore.".format(
                source_path
            )
        )


if __name__ == "__main__":
    args = create_parser().parse_args()
    setup_logging(args)
    compress_mags(args.source_path, args.layer_name, args.target_path, args.mag, args)
