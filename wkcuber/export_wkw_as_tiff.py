from argparse import ArgumentParser

import logging
import wkw
import os
import copy
from math import ceil
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Union

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
        "--tiling",
        "-t",
        help="In order to be able to convert large datasets it needs to be done in "
        "smaller pieces even in just one slice. Therefore this option will "
        'generate images per slice. The format is e.g. "x,y" when the axis is z',
        default=None,
    )

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

    parser.add_argument(
        "--tiling_slice_size",
        "-tiling_size",
        help="This will activate tiling. Thus each z-slice will be divided"
        "into multiple smaller tiff-pieces. The input is interpreted as the "
        "x and y size of a single tile with the format x,y.",
        default=None,
    )

    add_verbose_flag(parser)
    add_distribution_flags(parser)

    return parser


def wkw_name_and_bbox_to_tiff_name(name: str, slice_index: int) -> str:
    if name is None or name == "":
        return f"{slice_index}.tiff"
    else:
        return f"name_{slice_index}.tiff"


def calculate_tiling_size(
    bbox: Dict[str, Tuple[int, int, int]], tiling_size_x: int, tiling_size_y: int
):
    tiling_number_x = ceil(bbox["size"][0] / tiling_size_x)
    tiling_number_y = ceil(bbox["size"][1] / tiling_size_y)
    return (tiling_number_x, tiling_number_y)


def wkw_slice_to_image(data_slice: np.ndarray):
    # discard the z dimension
    data_slice = data_slice.squeeze(axis=3)
    if data_slice.shape[0] == 1:
        # discard greyscale dimension
        data_slice = data_slice.squeeze(axis=0)
        # swap the axis
        data_slice = data_slice.transpose((1, 0))
    else:
        # swap axis and move the channel axis
        data_slice = data_slice.transpose((2, 1, 0))
    return Image.fromarray(data_slice)


def export_tiff_slice(
    export_args: Tuple[
        int,
        Tuple[
            Dict[str, Tuple[int, int, int]], str, str, str,
            Union[None, Tuple[int, int]]
        ],
    ]
):
    slice_number, (tiff_bbox, dest_path, name, dataset_path, tiling_size) = export_args
    tiff_bbox["topleft"] = [
        tiff_bbox["topleft"][0],
        tiff_bbox["topleft"][1],
        tiff_bbox["topleft"][2] + slice_number,
    ]
    tiff_bbox["size"] = [tiff_bbox["size"][0], tiff_bbox["size"][1], 1]
    if tiling_size is None:
        with wkw.Dataset.open(dataset_path) as dataset:
            tiff_data = dataset.read(tiff_bbox["topleft"], tiff_bbox["size"])
        tiff_file_name = wkw_name_and_bbox_to_tiff_name(name, slice_number)

        tiff_file_path = os.path.join(dest_path, tiff_file_name)

        logging.info(f"saving slice {slice_number}")

        image = wkw_slice_to_image(tiff_data)
        image.save(tiff_file_path)
    else:
        tile_bbox = copy.deepcopy(tiff_bbox)
        tile_bbox["size"] = [tiling_size[0], tiling_size[1], 1]
        with wkw.Dataset.open(dataset_path) as dataset:
            for y_tile_index in range(ceil((tiff_bbox["size"][1] - 1) / tiling_size[1])):
                tile_bbox["topleft"][1] += tiling_size[1]
                tile_tiff_path = os.path.join(dest_path, str(slice_number), str(y_tile_index))
                os.makedirs(tile_tiff_path, exist_ok=True)
                for x_tile_index in range(ceil((tiff_bbox["size"][0] - 1) / tiling_size[0])):
                    print(tile_bbox)
                    tile_tiff_filename = f"{x_tile_index}.tiff"
                    tile_bbox["topleft"][0] += tiling_size[0]

                    tile_tiff_data = dataset.read(
                        tile_bbox["topleft"], tile_bbox["size"]
                    )
                    tile_image = wkw_slice_to_image(tile_tiff_data)
                    tile_image.save(os.path.join(tile_tiff_path, tile_tiff_filename))

                tile_bbox["topleft"][0] = tiff_bbox["topleft"][0]

        logging.info(f"saved all tiles of slice {slice_number}")


def export_tiff_stack(
    wkw_file_path, wkw_layer, bbox, mag, destination_path, name, tiling_slice_size, args
):
    os.makedirs(destination_path, exist_ok=True)

    dataset_path = os.path.join(wkw_file_path, wkw_layer, mag.to_layer_name())
    with get_executor_for_args(args) as executor:
        num_slices = bbox["size"][2]
        slices = range(num_slices)
        export_args = zip(
            slices,
            [(bbox, destination_path, name, dataset_path, tiling_slice_size)]
            * num_slices,
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

    if args.tiling_slice_size is not None:
        args.tiling_slice_size = [
            int(s.strip()) for s in args.tiling_slice_size.split(",")
        ]
        assert len(args.tiling_slice_size) == 2
        logging.info(
            f"Using tiling with the size of {args.tiling_slice_size[0]},{args.tiling_slice_size[1]}."
        )

    export_tiff_stack(
        wkw_file_path=args.source_path,
        wkw_layer=args.layer_name,
        bbox=bbox,
        mag=Mag(args.mag),
        destination_path=args.destination_path,
        name=args.name,
        tiling_slice_size=args.tiling_slice_size,
        args=args,
    )
