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
        "--source_path", "-s", help="Directory containing the wkw file(s).", required=True
    )

    parser.add_argument(
        "--destination_path",
        "-d",
        help="Output directory for the generated nifti files.",
        required=True,
    )

    parser.add_argument("--name", "-n", help="Name of the nifti", default="")

    parser.add_argument(
        "--bbox",
        help="The bounding box of which the nifti file should be generated."
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
        "--original_bbox",
        help="Easy way to add padding to the generated file. Format is: x,y,z,width,height,depth"
        "x,y,z is the offset of the wkw layer into the final bounding box; "
        "width,height,depth corresponds to final dimensions",
        default=None,
        type=parse_bounding_box,
    )

    parser.add_argument(
        "--bounding_box_crop",
        help="Easy way to crop the final file. Is applied AFTER everything else (including padding)."
        "Format is: x,y,z,width,height,depth",
        default=None,
        type=parse_bounding_box,
    )

    add_verbose_flag(parser)
    add_distribution_flags(parser)

    return parser


def export_layer_to_nifti(
    wkw_file_path: Path,
    bbox: Dict,
    mag: Mag,
    layer_name: str,
    destination_path: Path,
    name: str,
    original_bbox_size: Optional[List[int]] = None,
    offset_into_orginal_bbox: Optional[List[int]] = None,
    bounding_box_crop: Optional[List[int]] = None,
) -> None:
    wk_ds = Dataset(wkw_file_path)
    layer = wk_ds.get_layer(layer_name)
    mag_layer = layer.get_mag(mag)

    is_segmentation_layer = layer.category == LayerCategories.SEGMENTATION_TYPE

    data = mag_layer.read(bbox["topleft"], bbox["size"])
    data = data.transpose(1, 2, 3, 0)
    logging.info(f"Shape with layer {data.shape}")

    data = np.array(data)
    if is_segmentation_layer and data.max() > 0:
        factor = np.iinfo("uint8").max / data.max()
        data = data * factor
        data = data.astype(np.dtype("uint8"))

    if original_bbox_size and offset_into_orginal_bbox:
        padding_per_axis = []

        for i in range(3):
            if original_bbox_size[i] == data.shape[i]:
                padding_per_axis.append((0, 0))
            else:
                left_pad = offset_into_orginal_bbox[i]
                right_pad = (
                    original_bbox_size[i] - offset_into_orginal_bbox[i] - data.shape[i]
                )
                padding_per_axis.append((left_pad, right_pad))

        padding_per_axis.append((0, 0))
        print(padding_per_axis)
        data = np.pad(data, padding_per_axis, mode="constant", constant_values=0)

        if bounding_box_crop is not None:
            assert (
                len(bounding_box_crop) == 6
            ), "bounding box crop needs to have 6 parameters"

            print(f"Using Bounding Box {bounding_box_crop}")

            data = data[
                bounding_box_crop[0] : bounding_box_crop[0] + bounding_box_crop[3],
                bounding_box_crop[1] : bounding_box_crop[1] + bounding_box_crop[4],
                bounding_box_crop[2] : bounding_box_crop[2] + bounding_box_crop[5],
            ]

    img = nib.Nifti1Image(data, np.eye(4))

    destination_file = str(destination_path.joinpath(name + ".nii"))

    logging.info(f"Writing to {destination_file} with shape {data.shape}")

    nib.save(img, destination_file)


def export_nifti(
    wkw_file_path: Path,
    bbox: Optional[BoundingBox],
    mag: Mag,
    destination_path: Path,
    name: str,
    original_bbox_size: Optional[List[int]] = None,
    offset_into_orginal_bbox: Optional[List[int]] = None,
    bounding_box_crop: Optional[List[int]] = None,
) -> None:
    wk_ds = Dataset(wkw_file_path)

    for layer_name, layer in wk_ds.layers.items():
        if bbox is None:
            bbox_dict = {
                "topleft": layer.bounding_box.topleft,
                "size": layer.bounding_box.size,
            }
        else:
            bbox_dict = {"topleft": list(bbox.topleft), "size": list(bbox.size)}

        logging.info(f"Starting nifti export for bounding box: {bbox}")

        export_layer_to_nifti(
            wkw_file_path,
            bbox_dict,
            mag,
            layer_name,
            destination_path,
            name + "_" + layer_name,
            original_bbox_size,
            offset_into_orginal_bbox,
            bounding_box_crop,
        )


def export_wkw_as_nifti(args: Namespace) -> None:
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    original_bbox_size = args.original_bbox.size if args.original_bbox else None
    offset_into_orginal_bbox = (
        args.original_bbox.topleft if args.original_bbox else None
    )

    export_nifti(
        wkw_file_path=Path(args.source_path),
        bbox=args.bbox,
        mag=Mag(args.mag),
        destination_path=Path(args.destination_path),
        name=args.name,
        original_bbox_size=original_bbox_size,
        offset_into_orginal_bbox=offset_into_orginal_bbox,
        bounding_box_crop=args.bounding_box_crop,
    )


def export_wkw_as_nifti_from_arg_list(arg_list: List = None) -> None:
    parsed_args = create_parser().parse_args(arg_list)
    setup_logging(parsed_args)
    export_wkw_as_nifti(parsed_args)


if __name__ == "__main__":
    export_wkw_as_nifti_from_arg_list()
