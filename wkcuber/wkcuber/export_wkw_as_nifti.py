import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np

from wkcuber.api.bounding_box import BoundingBox
from wkcuber.api.dataset import Dataset, LayerCategories
from wkcuber.mag import Mag
from wkcuber.utils import (
    add_distribution_flags,
    add_verbose_flag,
    parse_bounding_box,
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
        "--padded_bbox",
        help="After reading from --source_bbox in the input file, the data is zero-padded to fit --padded_bbox. --source_bbox must be contained in --padded_bbox. Format is: x,y,z,width,height,depth"
        "x,y,z is the offset of the wkw layer into the final bounding box; "
        "width,height,depth corresponds to final dimensions",
        default=None,
        type=parse_bounding_box,
    )

    parser.add_argument(
        "--cropping_bbox",
        help="After reading (and potentially padding) the input data, the output is cropped according to --cropping_bbox. --cropping_bbox must be contained in --padded_bbox"
        "Format is: x,y,z,width,height,depth",
        default=None,
        type=parse_bounding_box,
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
    padded_bbox: Optional[BoundingBox] = None,
    cropping_bbox: Optional[BoundingBox] = None,
) -> None:
    dataset = Dataset(wkw_file_path)
    layer = dataset.get_layer(layer_name)
    mag_layer = layer.get_mag(mag)

    is_segmentation_layer = layer.category == LayerCategories.SEGMENTATION_TYPE

    data = mag_layer.read(source_bbox)
    data = data.transpose(1, 2, 3, 0)
    logging.info(f"Shape with layer {data.shape}")

    data = np.array(data)
    if is_segmentation_layer and data.max() > 0:
        factor = np.iinfo("uint8").max / data.max()
        data = data * factor
        data = data.astype(np.dtype("uint8"))

    if padded_bbox:
        data_bbox_relative_to_pad_bbox = BoundingBox(
            padded_bbox.topleft, data.shape[:3]
        )
        pad_bbox_as_origin = BoundingBox((0, 0, 0), padded_bbox.size)

        assert pad_bbox_as_origin.contains_bbox(
            data_bbox_relative_to_pad_bbox
        ), "padded_bbox should contain source_bbox"

        padding_per_axis = []

        for i in range(3):
            if padded_bbox.size[i] == data.shape[i]:
                padding_per_axis.append((0, 0))
            else:
                left_pad = data_bbox_relative_to_pad_bbox.topleft[i]
                right_pad = (
                    pad_bbox_as_origin.size[i]
                    - data_bbox_relative_to_pad_bbox.bottomright[i]
                )
                padding_per_axis.append((left_pad, right_pad))

        padding_per_axis.append((0, 0))
        data = np.pad(data, padding_per_axis, mode="constant", constant_values=0)

        if cropping_bbox is not None:
            assert pad_bbox_as_origin.contains_bbox(
                cropping_bbox
            ), "padded_bbox should contain cropping_bbox"

            logging.info(f"Using Bounding Box {cropping_bbox}")

            data = data[
                cropping_bbox[0] : cropping_bbox[0] + cropping_bbox[3],
                cropping_bbox[1] : cropping_bbox[1] + cropping_bbox[4],
                cropping_bbox[2] : cropping_bbox[2] + cropping_bbox[5],
            ]

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
    padded_bbox: Optional[BoundingBox] = None,
    cropping_bbox: Optional[BoundingBox] = None,
) -> None:
    dataset = Dataset(wkw_file_path)

    for layer_name, layer in dataset.layers.items():
        logging.info(f"Starting nifti export for bounding box: {source_bbox}")

        export_layer_to_nifti(
            wkw_file_path,
            layer.bounding_box if source_bbox is None else source_bbox,
            mag,
            layer_name,
            destination_path,
            name + "_" + layer_name,
            padded_bbox,
            cropping_bbox,
        )


def export_wkw_as_nifti(args: Namespace) -> None:
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    export_nifti(
        wkw_file_path=Path(args.source_path),
        source_bbox=args.source_bbox,
        mag=Mag(args.mag),
        destination_path=Path(args.destination_path),
        name=args.name,
        padded_bbox=args.padded_bbox,
        cropping_bbox=args.cropping_bbox,
    )


def export_wkw_as_nifti_from_arg_list(arg_list: List = None) -> None:
    parsed_args = create_parser().parse_args(arg_list)
    setup_logging(parsed_args)
    export_wkw_as_nifti(parsed_args)


if __name__ == "__main__":
    export_wkw_as_nifti_from_arg_list()
