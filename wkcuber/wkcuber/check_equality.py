import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from webknossos import Dataset, View

from ._internal.utils import (
    add_distribution_flags,
    add_verbose_flag,
    parse_path,
    setup_logging,
    setup_warnings,
)
from .compress import BACKUP_EXT


def create_parser() -> ArgumentParser:
    parser = ArgumentParser()

    parser.add_argument(
        "source_path", help="Path to input WKW dataset", type=parse_path
    )

    parser.add_argument(
        "target_path",
        help="WKW dataset with which to compare the input dataset.",
        type=parse_path,
    )

    parser.add_argument(
        "--layer_name",
        "-l",
        help="Name of the layer to compare (if not provided, all layers are compared)",
        default=None,
    )

    add_verbose_flag(parser)
    add_distribution_flags(parser)

    return parser


def assert_equality_job(args: Tuple[View, View, int]) -> None:
    source_view, target_view, _ = args

    assert np.all(
        source_view.read() == target_view.read()
    ), f"Data differs in bounding box {source_view.bounding_box}."


def check_equality(
    source_path: Path, target_path: Path, args: Optional[Namespace] = None
) -> None:

    logging.info(f"Comparing {source_path} with {target_path}")

    source_dataset = Dataset.open(source_path)
    target_dataset = Dataset.open(target_path)

    source_layer_names = set(source_dataset.layers.keys())
    target_layer_names = set(target_dataset.layers.keys())

    layer_names = list(source_layer_names)

    if args is not None and args.layer_name is not None:
        assert (
            args.layer_name in source_layer_names
        ), f"Provided layer {args.layer_name} does not exist in source dataset."
        assert (
            args.layer_name in target_layer_names
        ), f"Provided layer {args.layer_name} does not exist in target dataset."
        layer_names = [args.layer_name]

    else:
        assert (
            source_layer_names == target_layer_names
        ), f"The provided input datasets have different layers: {source_layer_names} != {target_layer_names}"

    for layer_name in layer_names:
        logging.info(f"Checking layer_name: {layer_name}")

        source_layer = source_dataset.layers[layer_name]
        target_layer = target_dataset.layers[layer_name]

        assert (
            source_layer.bounding_box == target_layer.bounding_box
        ), f"The bounding boxes of {source_path}/{layer_name} and {target_path}/{layer_name} are not equal: {source_layer.bounding_box} != {target_layer.bounding_box}"

        source_mags = set(source_layer.mags.keys())
        target_mags = set(target_layer.mags.keys())

        assert (
            source_mags == target_mags
        ), f"The mags of {source_path}/{layer_name} and {target_path}/{layer_name} are not equal: {source_mags} != {target_mags}"

        for mag in source_mags:
            source_mag = source_layer.mags[mag]
            target_mag = target_layer.mags[mag]

            logging.info(f"Start verification of {layer_name} in mag {mag}")
            assert source_mag.content_is_equal(target_mag, args)

    logging.info(
        f"The following datasets seem to be equal (with regard to the layers: {layer_names}):"
    )
    logging.info(source_path)
    logging.info(target_path)


if __name__ == "__main__":
    setup_warnings()
    args = create_parser().parse_args()
    setup_logging(args)

    if args.target_path is None:
        target_path = args.source_path.with_suffix(BACKUP_EXT)
    else:
        target_path = args.target_path
    check_equality(args.source_path, target_path, args)
