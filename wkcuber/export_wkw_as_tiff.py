from argparse import ArgumentParser

import logging
import wkw
import os
import numpy as np
from PIL import Image
from typing import Tuple, Dict

from .metadata import read_metadata_for_layer
from .utils import add_verbose_flag, add_distribution_flags, get_executor_for_args
from .mag import Mag


def create_parser():
    parser = ArgumentParser()

    parser.add_argument(
        "--source_path", "-s", help="Directory containing the wkw file.", required=True
    )

    parser.add_argument(
        "--destination_path",
        "-d",
        help="Output directory for the generated tiff files.",
        required=True,
    )

    parser.add_argument(
        "--layer_name",
        "-l",
        help="Name of the layer that will be converted to a tiff stack",
        default="color",
    )

    parser.add_argument("--name", "-n", help="Name of the tiffs", default="")

    parser.add_argument(
        "--bbox",
        "-b",
        help="The BoundingBox of which the tiff stack should be generated."
        "The input format is x,y,z,width,height,depth."
        "(By default, data for the full bounding box of the dataset is generated)",
        default=None,
    )

    parser.add_argument(
        "--mag", "-m", help="The magnification that should be read", default=1
    )

    add_verbose_flag(parser)
    add_distribution_flags(parser)

    return parser


def wkw_name_and_bbox_to_tiff_name(name: str, slice_index: int) -> str:
    if name is None or name == "":
        return f"{slice_index}.tiff"
    else:
        return f"name_{slice_index}.tiff"


def export_tiff_slice(
    export_args: Tuple[int, Tuple[Dict[str, Tuple[int, int, int]], str, str, str]]
):
    slice_number, (bbox, dest_path, name, dataset_path) = export_args
    tiff_bbox = bbox
    tiff_bbox["topleft"] = [
        tiff_bbox["topleft"][0],
        tiff_bbox["topleft"][1],
        tiff_bbox["topleft"][2] + slice_number,
    ]
    tiff_bbox["size"] = [tiff_bbox["size"][0], tiff_bbox["size"][1], 1]

    tiff_file_name = wkw_name_and_bbox_to_tiff_name(name, slice_number)

    tiff_file_path = os.path.join(dest_path, tiff_file_name)

    with wkw.Dataset.open(dataset_path) as dataset:
        tiff_data = dataset.read(tiff_bbox["topleft"], tiff_bbox["size"])

    # discard the z dimension
    tiff_data.squeeze(axis=3)
    if tiff_data.shape[0] == 1:
        # discard greyscale dimension
        tiff_data.squeeze(axis=0)
        # swap the axis
        tiff_data.transpose((1, 0))
    else:
        # swap axis and move the channel axis
        tiff_data.transpose((2, 1, 0))

    logging.info(f"saving slice {slice_number}")

    image = Image.fromarray(tiff_data)
    image.save(tiff_file_path)


def export_tiff_stack(
    wkw_file_path, wkw_layer, bbox, mag, destination_path, name, args
):
    os.makedirs(destination_path, exist_ok=True)

    dataset_path = os.path.join(wkw_file_path, wkw_layer, mag.to_layer_name())
    with get_executor_for_args(args) as executor:
        num_slices = bbox["size"][2]
        slices = range(num_slices)
        export_args = zip(
            slices, [(bbox, destination_path, name, dataset_path)] * num_slices
        )
        logging.info(f"starting jobs")
        executor.map(export_tiff_slice, export_args)


if __name__ == "__main__":
    args = create_parser().parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    if args.bbox is None:
        _, _, bbox, _ = read_metadata_for_layer(args.source_path, args.layer_name)
    else:
        bbox = [int(s.strip()) for s in args.bbox.split(",")]
        assert len(bbox) == 6
        bbox = {"topleft": bbox[0:3], "size": bbox[3:6]}

    logging.info(f"Starting tiff export for bounding box: {bbox}")

    export_tiff_stack(
        wkw_file_path=args.source_path,
        wkw_layer=args.layer_name,
        bbox=bbox,
        mag=Mag(args.mag),
        destination_path=args.destination_path,
        name=args.name,
        args=args,
    )
