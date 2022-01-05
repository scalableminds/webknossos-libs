import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Callable, Optional

from wkcuber.api.dataset import Dataset
from webknossos.geometry import BoundingBox, Vec3Int
import numpy as np

from wkcuber.mag import Mag
from .utils import (
    add_verbose_flag,
    open_wkw,
    WkwDatasetInfo,
    add_distribution_flags,
    get_executor_for_args,
    wait_and_ensure_success,
    setup_logging,
)
from .metadata import detect_resolutions, detect_bbox, detect_layers
import functools
from .compress import BACKUP_EXT


CHUNK_SIZE = Vec3Int.full(1024)


def named_partial(func: Callable, *args: Any, **kwargs: Any) -> Callable:
    # Propagate __name__ and __doc__ attributes to partial function
    partial_func = functools.partial(func, *args, **kwargs)
    functools.update_wrapper(partial_func, func)
    if hasattr(func, "__annotations__"):
        # Generic types cannot be pickled in Python <= 3.6, see https://github.com/python/typing/issues/511
        partial_func.__annotations__ = {}
    return partial_func


def create_parser() -> ArgumentParser:
    parser = ArgumentParser()

    parser.add_argument("source_path", help="Path to input WKW dataset", type=Path)

    parser.add_argument(
        "target_path",
        help="WKW dataset with which to compare the input dataset.",
        type=Path,
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


def assert_equality_for_chunk(
    source_path: Path,
    target_path: Path,
    layer_name: str,
    mag: Mag,
    sub_box: BoundingBox,
) -> None:
    wk_dataset = Dataset.open(source_path)
    layer = wk_dataset.get_layer(layer_name)
    backup_wkw_info = WkwDatasetInfo(target_path, layer_name, mag, header=None)
    with open_wkw(backup_wkw_info) as backup_wkw:
        mag_view = layer.get_mag(mag)
        logging.info(f"Checking sub_box: {sub_box}")

        data = mag_view.read(sub_box.topleft, sub_box.size)
        backup_data = backup_wkw.read(sub_box.topleft, sub_box.size)
        assert np.all(
            data == backup_data
        ), f"Data differs in bounding box {sub_box} for layer {layer_name} with mag {mag}"


def check_equality(
    source_path: Path, target_path: Path, args: Optional[Namespace] = None
) -> None:

    logging.info(f"Comparing {source_path} with {target_path}")

    wk_src_dataset = Dataset.open(source_path)
    src_layer_names = list(wk_src_dataset.layers.keys())
    target_layer_names = [
        layer["name"] for layer in detect_layers(target_path, 0, False)
    ]
    assert set(src_layer_names) == set(
        target_layer_names
    ), f"The provided input datasets have different layers: {src_layer_names} != {target_layer_names}"

    existing_layer_names = src_layer_names

    if args is not None and args.layer_name is not None:
        assert (
            args.layer_name in existing_layer_names
        ), f"Provided layer {args.layer_name} does not exist in input dataset."
        existing_layer_names = [args.layer_name]

    for layer_name in existing_layer_names:

        logging.info(f"Checking layer_name: {layer_name}")

        source_mags = list(detect_resolutions(source_path, layer_name))
        target_mags = list(detect_resolutions(target_path, layer_name))
        source_mags.sort()
        target_mags.sort()
        mags = source_mags

        assert (
            source_mags == target_mags
        ), f"The mags between {source_path}/{layer_name} and {target_path}/{layer_name} are not equal: {source_mags} != {target_mags}"

        official_bbox = wk_src_dataset.get_layer(layer_name).bounding_box

        for mag in mags:
            inferred_src_bbox_dict = detect_bbox(source_path, layer_name, mag)
            inferred_target_bbox_dict = detect_bbox(target_path, layer_name, mag)
            assert inferred_src_bbox_dict is not None
            assert inferred_target_bbox_dict is not None
            inferred_src_bbox = BoundingBox.from_auto(inferred_src_bbox_dict)
            inferred_target_bbox = BoundingBox.from_auto(inferred_target_bbox_dict)

            bbox = inferred_src_bbox.extended_by(inferred_target_bbox).extended_by(
                official_bbox
            )
            logging.info(f"Start verification of {layer_name} in mag {mag} in {bbox}")

            with get_executor_for_args(args) as executor:
                boxes = list(bbox.chunk(CHUNK_SIZE, CHUNK_SIZE))
                assert_fn = named_partial(
                    assert_equality_for_chunk, source_path, target_path, layer_name, mag
                )

                wait_and_ensure_success(executor.map_to_futures(assert_fn, boxes))

    logging.info(
        f"The following datasets seem to be equal (with regard to the layers: {existing_layer_names}):"
    )
    logging.info(source_path)
    logging.info(target_path)


if __name__ == "__main__":
    args = create_parser().parse_args()
    setup_logging(args)

    if args.target_path is None:
        target_path = args.source_path + BACKUP_EXT
    else:
        target_path = args.target_path
    check_equality(args.source_path, target_path, args)
