import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from shutil import rmtree
from typing import List, Optional

from wkcuber.api.dataset import Dataset

from .mag import Mag
from .utils import add_distribution_flags, add_verbose_flag, setup_logging

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
    args: Optional[Namespace] = None,
) -> None:
    Dataset.open(source_path).get_layer(layer_name).get_mag(mag).compress(
        target_path=Path(target_path), args=args
    )


def compress_mag_inplace(
    target_path: Path, layer_name: str, mag: Mag, args: Optional[Namespace] = None
) -> None:
    Dataset.open(target_path).get_layer(layer_name).get_mag(mag).compress(args=args)


def compress_mags(
    source_path: Path,
    layer_name: str,
    target_path: Optional[Path] = None,
    mags: Optional[List[Mag]] = None,
    args: Optional[Namespace] = None,
) -> None:
    if target_path is None:
        target = source_path.with_suffix(".tmp")
    else:
        target = target_path

    layer = Dataset.open(source_path).get_layer(layer_name)
    if mags is None:
        mags = list(layer.mags.keys())

    for mag, mag_view in Dataset.open(source_path).get_layer(layer_name).mags.items():
        if mag in mags:
            mag_view.compress(target_path=Path(target), args=args)

    if target_path is None:
        backup_dir = source_path.with_suffix(BACKUP_EXT)
        (backup_dir / layer_name).mkdir(parents=True, exist_ok=True)
        for mag in mags:
            (source_path / layer_name / str(mag)).rename(
                (backup_dir / layer_name / str(mag))
            )

            (target / layer_name / str(mag)).rename(
                str(source_path / layer_name / str(mag)),
            )
        rmtree(target)
        logging.info(
            "Old files are still present in '{0}.bak'. Please remove them when not required anymore.".format(
                source_path
            )
        )


if __name__ == "__main__":
    args = create_parser().parse_args()
    setup_logging(args)
    compress_mags(args.source_path, args.layer_name, args.target_path, args.mag, args)
