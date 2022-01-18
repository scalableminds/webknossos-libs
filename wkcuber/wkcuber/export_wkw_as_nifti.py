import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List, Optional, Tuple

import nibabel as nib
import numpy as np

from wkcuber.api.bounding_box import BoundingBox
from wkcuber.api.dataset import Dataset
from wkcuber.api.layer_categories import SEGMENTATION_CATEGORY
from wkcuber.mag import Mag
from wkcuber.utils import (
    add_distribution_flags,
    add_verbose_flag,
    parse_bounding_box,
    parse_padding,
    setup_logging,
)


def create_parser() -> ArgumentParser:
    parser = ArgumentParser()

    parser.add_argument(
        "--source_path",
        "-s",
        help="Directory containing the wkw file(s).",
        required=True,
    )

    parser.add_argument(
        "--destination_path",
        "-d",
        help="Output directory for the generated nifti files. One file will be generated per wkw layer.",
        required=True,
    )

    parser.add_argument("--name", "-n", help="Name of the nifti", default="")

    parser.add_argument(
        "--source_bbox",
        help="The bounding box in the nifti file from which data is read."
        "The input format is x,y,z,width,height,depth."
        "(By default, data for the full bounding box of the dataset is generated)",
        default=None,
        type=parse_bounding_box,
    )

    parser.add_argument(
        "--mag", "-m", help="The magnification that should be read", default=1
    )

    parser.add_argument(
        "--downsample", help="Downsample each nifti image", default=1, type=int
    )

    parser.add_argument(
        "--padding",
        help="After reading from --source_bbox in the input file, the data is zero-padded according to padding."
        "Format is left_pad_x,left_pad_y,left_pad_z,right_pad_x,right_pad_y,right_pad_z",
        default=None,
        type=parse_padding,
    )

    add_verbose_flag(parser)
    add_distribution_flags(parser)

    return parser


def export_layer_to_nifti(
    wkw_file_path: Path,
    source_bbox: BoundingBox,
    mag: Mag,
    layer_name: str,
    destination_path: Path,
    name: str,
    padding: Optional[Tuple[int, ...]] = None,
) -> None:
    dataset = Dataset.open(wkw_file_path)
    layer = dataset.get_layer(layer_name)
    mag_layer = layer.get_mag(mag)

    is_segmentation_layer = layer.category == SEGMENTATION_CATEGORY

    data = mag_layer.read(source_bbox.topleft, source_bbox.size)
    data = data.transpose(1, 2, 3, 0)
    logging.info(f"Shape with layer {data.shape}")

    data = np.array(data)
    if is_segmentation_layer and data.max() > 0:
        factor = np.iinfo("uint8").max / data.max()
        data = data * factor
        data = data.astype(np.dtype("uint8"))

    if padding:
        assert len(padding) == 6, "padding needs 6 values"

        padding_per_axis = list(zip(padding[:3], padding[3:]))
        padding_per_axis.append((0, 0))
        data = np.pad(data, padding_per_axis, mode="constant", constant_values=0)

    img = nib.Nifti1Image(data, np.eye(4))

    destination_file = str(destination_path.joinpath(name + ".nii"))

    logging.info(f"Writing to {destination_file} with shape {data.shape}")
    nib.save(img, destination_file)


def export_nifti(
    wkw_file_path: Path,
    source_bbox: Optional[BoundingBox],
    mag: Mag,
    destination_path: Path,
    name: str,
    padding: Optional[Tuple[int, ...]] = None,
) -> None:
    dataset = Dataset.open(wkw_file_path)

    for layer_name, layer in dataset.layers.items():
        logging.info(f"Starting nifti export for bounding box: {source_bbox}")

        export_layer_to_nifti(
            wkw_file_path,
            layer.bounding_box if source_bbox is None else source_bbox,
            mag,
            layer_name,
            destination_path,
            name + "_" + layer_name,
            padding,
        )


def export_wkw_as_nifti(args: Namespace) -> None:
    setup_logging(args)

    export_nifti(
        wkw_file_path=Path(args.source_path),
        source_bbox=args.source_bbox,
        mag=Mag(args.mag),
        destination_path=Path(args.destination_path),
        name=args.name,
        padding=args.padding,
    )


def export_wkw_as_nifti_from_arg_list(arg_list: Optional[List] = None) -> None:
    parsed_args = create_parser().parse_args(arg_list)
    export_wkw_as_nifti(parsed_args)


if __name__ == "__main__":
    export_wkw_as_nifti_from_arg_list()
