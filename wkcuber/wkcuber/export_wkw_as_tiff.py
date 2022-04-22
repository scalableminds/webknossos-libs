import logging
from argparse import ArgumentParser, Namespace
from functools import partial
from math import ceil
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
from PIL import Image
from scipy.ndimage.interpolation import zoom
from webknossos import BoundingBox, Dataset, MagView
from webknossos.dataset.view import View
from webknossos.geometry.vec3_int import Vec3Int
from webknossos.utils import wait_and_ensure_success

from ._internal.utils import (
    add_batch_size_flag,
    add_distribution_flags,
    add_verbose_flag,
    get_executor_for_args,
    parse_bounding_box,
    parse_path,
    setup_logging,
    setup_warnings,
)


def create_parser() -> ArgumentParser:
    parser = ArgumentParser()

    parser.add_argument(
        "--source_path",
        "-s",
        help="Directory containing the wkw file.",
        required=True,
        type=parse_path,
    )

    parser.add_argument(
        "--destination_path",
        "-d",
        help="Output directory for the generated tiff files.",
        required=True,
        type=Path,
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
        help="The BoundingBox of which the tiff stack should be generated."
        "The input format is x,y,z,width,height,depth."
        "(By default, data for the full bounding box of the dataset is generated)",
        default=None,
        type=parse_bounding_box,
    )

    parser.add_argument(
        "--mag", "-m", help="The magnification that should be read", default=1
    )

    parser.add_argument(
        "--downsample", help="Downsample each tiff image", default=1, type=int
    )

    tiling_option_group = parser.add_mutually_exclusive_group()

    tiling_option_group.add_argument(
        "--tiles_per_dimension",
        "-t",
        help='For very large datasets, it is recommended to enable tiling which will ensure that each slice is exported to multiple images (i.e., tiles). As a parameter you should provide the amount of tiles per dimension in the form of "x,y".'
        'Also see at "--tile_size" to specify the absolute size of the tiles.',
        default=None,
    )

    tiling_option_group.add_argument(
        "--tile_size",
        help='For very large datasets, it is recommended to enable tiling which will ensure that each slice is exported to multiple images (i.e., tiles). As a parameter you should provide the the size of each tile per dimension in the form of "x,y".'
        'Also see at "--tiles_per_dimension" to specify the number of tiles in the dimensions.',
        default=None,
    )

    add_batch_size_flag(parser)

    add_verbose_flag(parser)
    add_distribution_flags(parser)

    return parser


def _make_tiff_name(name: str, slice_index: int) -> str:
    if name is None or name == "":
        return f"{slice_index:06d}.tiff"
    else:
        return f"{name}_{slice_index:06d}.tiff"


def _slice_to_image(data_slice: np.ndarray, downsample: int = 1) -> Image:
    if data_slice.shape[0] == 1:
        # discard greyscale dimension
        data_slice = data_slice.squeeze(axis=0)
        # swap the axis
        data_slice = data_slice.transpose((1, 0))
    else:
        # swap axis and move the channel axis
        data_slice = data_slice.transpose((2, 1, 0))

    if downsample > 1:
        data_slice = zoom(
            data_slice,
            1 / downsample,
            output=data_slice.dtype,
            order=1,
            # this does not mean nearest interpolation,
            # it corresponds to how the borders are treated.
            mode="nearest",
            prefilter=True,
        )
    return Image.fromarray(data_slice)


def export_tiff_slice(
    dest_path: Path,
    name: str,
    tiling_size: Union[None, Tuple[int, int]],
    downsample: int,
    start_slice_index: int,
    view: View,
) -> None:
    tiff_bbox_mag1 = view.bounding_box
    tiff_bbox = tiff_bbox_mag1.in_mag(view.mag)

    if tiling_size is None:
        tiff_data = view.read()
    else:
        padded_tiff_bbox_size = Vec3Int(
            tiling_size[0] * ceil(tiff_bbox.size.x / tiling_size[0]),
            tiling_size[1] * ceil(tiff_bbox.size.y / tiling_size[1]),
            tiff_bbox.size.z,
        )
        tiff_data = view.read()
        padded_tiff_data = np.zeros(
            (tiff_data.shape[0],) + padded_tiff_bbox_size.to_tuple(),
            dtype=tiff_data.dtype,
        )
        padded_tiff_data[
            :, 0 : tiff_data.shape[1], 0 : tiff_data.shape[2], 0 : tiff_data.shape[3]
        ] = tiff_data
        tiff_data = padded_tiff_data
    for slice_index in range(tiff_bbox.size.z):
        slice_name_number = (
            tiff_bbox_mag1.topleft.z + slice_index + 1 - start_slice_index
        )
        if tiling_size is None:
            tiff_file_name = _make_tiff_name(name, slice_name_number)
            tiff_file_path = dest_path / tiff_file_name

            image = _slice_to_image(tiff_data[:, :, :, slice_index], downsample)
            image.save(tiff_file_path)
            logging.debug("Saved slice %s", slice_name_number)

        else:
            for y_tile_index in range(ceil(tiff_bbox.size.y / tiling_size[1])):
                tile_tiff_path = (
                    dest_path / str(slice_name_number) / str(y_tile_index + 1)
                )
                tile_tiff_path.mkdir(parents=True, exist_ok=True)
                for x_tile_index in range(ceil(tiff_bbox.size.x / tiling_size[0])):
                    tile_tiff_filename = f"{x_tile_index + 1}.tiff"
                    tile_image = _slice_to_image(
                        tiff_data[
                            :,
                            x_tile_index
                            * tiling_size[0] : (x_tile_index + 1)
                            * tiling_size[0],
                            y_tile_index
                            * tiling_size[1] : (y_tile_index + 1)
                            * tiling_size[1],
                            slice_index,
                        ],
                        downsample,
                    )

                    tile_image.save(tile_tiff_path / tile_tiff_filename)

            logging.debug(f"Saved tiles for slice {slice_name_number}")

    logging.debug(f"Saved all tiles of bbox {tiff_bbox}")


def export_tiff_stack(
    mag_view: MagView,
    bbox: BoundingBox,
    destination_path: Path,
    name: str,
    tiling_slice_size: Union[None, Tuple[int, int]],
    batch_size: int,
    downsample: int,
    args: Namespace,
) -> None:
    destination_path.mkdir(parents=True, exist_ok=True)

    with get_executor_for_args(args) as executor:
        view = mag_view.get_view(absolute_offset=bbox.topleft, size=bbox.size)

        view_chunks = [
            view.get_view(
                relative_offset=(0, 0, z),
                size=(bbox.size.x, bbox.size.y, min(batch_size, bbox.size.z - z)),
            )
            for z in range(0, bbox.size.z, batch_size)
        ]

        wait_and_ensure_success(
            executor.map_to_futures(
                partial(
                    export_tiff_slice,
                    destination_path,
                    name,
                    tiling_slice_size,
                    downsample,
                    bbox.topleft.z,
                ),
                view_chunks,
            ),
            progress_desc="Exporting tiff files",
        )


def export_wkw_as_tiff(args: Namespace) -> None:
    setup_logging(args)

    mag_view = (
        Dataset.open(args.source_path).get_layer(args.layer_name).get_mag(args.mag)
    )

    bbox = mag_view.bounding_box if args.bbox is None else args.bbox

    logging.info(f"Starting tiff export for bounding box: {bbox}")

    if args.tiles_per_dimension is not None:
        args.tile_size = [int(s.strip()) for s in args.tiles_per_dimension.split(",")]
        assert len(args.tile_size) == 2
        logging.info(
            f"Using tiling with {args.tile_size[0]},{args.tile_size[1]} tiles in the dimensions."
        )
        args.tile_size[0] = ceil(bbox.in_mag(mag_view.mag).size.x / args.tile_size[0])
        args.tile_size[1] = ceil(bbox.in_mag(mag_view.mag).size.y / args.tile_size[1])

    elif args.tile_size is not None:
        args.tile_size = [int(s.strip()) for s in args.tile_size.split(",")]
        assert len(args.tile_size) == 2
        logging.info(
            f"Using tiling with the size of {args.tile_size[0]},{args.tile_size[1]}."
        )
    args.batch_size = int(args.batch_size)

    export_tiff_stack(
        mag_view=mag_view,
        bbox=bbox,
        destination_path=args.destination_path,
        name=args.name,
        tiling_slice_size=args.tile_size,
        batch_size=args.batch_size,
        downsample=args.downsample,
        args=args,
    )


def run(args_list: List) -> None:
    arguments = create_parser().parse_args(args_list)
    export_wkw_as_tiff(arguments)


if __name__ == "__main__":
    setup_warnings()
    arguments = create_parser().parse_args()
    export_wkw_as_tiff(arguments)
