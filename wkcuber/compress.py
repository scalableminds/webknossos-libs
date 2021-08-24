from pathlib import Path

import shutil
import logging
from argparse import ArgumentParser, Namespace
from os import makedirs

from wkcuber.api.dataset import Dataset
from .mag import Mag

from .utils import add_verbose_flag, add_distribution_flags, setup_logging
from typing import List

BACKUP_EXT = ".bak"


def create_parser() -> ArgumentParser:
    parser = ArgumentParser()

    parser.add_argument(
        "source_path", help="Directory containing the source WKW dataset.", type=Path
    )

    parser.add_argument(
        "target_path",
        help="Output directory for the compressed WKW dataset.",
        nargs="?",
        default=None,
        type=Path,
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
    source_path: Path,
    layer_name: str,
    target_path: Path,
    mag: Mag,
    args: Namespace = None,
) -> None:
    Dataset(source_path).get_layer(layer_name).get_mag(mag).compress(
        target_path=Path(target_path), args=args
    )


def compress_mag_inplace(
    target_path: Path, layer_name: str, mag: Mag, args: Namespace = None
) -> None:
    Dataset(target_path).get_layer(layer_name).get_mag(mag).compress(args=args)


def compress_mags(
    source_path: Path,
    layer_name: str,
    target_path: Path = None,
    mags: List[Mag] = None,
    args: Namespace = None,
) -> None:
    if target_path is None:
        target = source_path.with_suffix(".tmp")
    else:
        target = target_path

    layer = Dataset(source_path).get_layer(layer_name)
    if mags is None:
        mags = list(layer.mags.keys())

    for mag, mag_view in Dataset(source_path).get_layer(layer_name).mags.items():
        if mag in mags:
            mag_view.compress(target_path=Path(target), args=args)

    if target_path is None:
        backup_dir = source_path.with_suffix(BACKUP_EXT)
        makedirs(backup_dir / layer_name, exist_ok=True)
        for mag in mags:
            shutil.move(
                str(source_path / layer_name / str(mag)),
                str(backup_dir / layer_name / str(mag)),
            )
            shutil.move(
                str(target / layer_name / str(mag)),
                str(source_path / layer_name / str(mag)),
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
