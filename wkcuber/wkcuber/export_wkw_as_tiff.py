from argparse import ArgumentParser, Namespace
from functools import partial
import logging
from pathlib import Path

import wkw
from math import ceil
import numpy as np
from PIL import Image
from scipy.ndimage.interpolation import zoom
from typing import Tuple, Dict, Union, List

from wkcuber.metadata import read_metadata_for_layer
from wkcuber.utils import (
    add_verbose_flag,
    add_distribution_flags,
    get_executor_for_args,
    add_batch_size_flag,
    parse_bounding_box,
    setup_logging,
)
from wkcuber.mag import Mag
from wkcuber.utils import wait_and_ensure_success


def create_parser() -> ArgumentParser:
    parser = ArgumentParser()

    parser.add_argument(
        "--source_path",
        "-s",
        help="Directory containing the wkw file.",
        required=True,
        type=Path,
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


def wkw_name_and_bbox_to_tiff_name(name: str, slice_index: int) -> str:
    if name is None or name == "":
        return f"{slice_index:06d}.tiff"
    else:
        return f"{name}_{slice_index:06d}.tiff"


def wkw_slice_to_image(data_slice: np.ndarray, downsample: int = 1) -> Image:
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
    tiff_bbox: Dict[str, Tuple[int, int, int]],
    dest_path: Path,
    name: str,
    dataset_path: Path,
    tiling_size: Union[None, Tuple[int, int]],
    batch_size: int,
    downsample: int,
    mag: Mag,
    batch_number: int,
) -> None:

    tiff_bbox = tiff_bbox.copy()
    number_of_slices = (
        min(tiff_bbox["size"][2] - batch_number * batch_size, batch_size) // mag.z
    )
    tiff_bbox["size"] = (
        tiff_bbox["size"][0] // mag.x,
        tiff_bbox["size"][1] // mag.y,
        number_of_slices,
    )
    tiff_bbox["topleft"] = (
        tiff_bbox["topleft"][0] // mag.x,
        tiff_bbox["topleft"][1] // mag.y,
        (tiff_bbox["topleft"][2] + batch_number * batch_size) // mag.z,
    )

    with wkw.Dataset.open(str(dataset_path)) as dataset:
        if tiling_size is None:
            tiff_data = dataset.read(tiff_bbox["topleft"], tiff_bbox["size"])
        else:
            padded_tiff_bbox_size = [
                tiling_size[0] * ceil(tiff_bbox["size"][0] / tiling_size[0]),
                tiling_size[1] * ceil(tiff_bbox["size"][1] / tiling_size[1]),
                number_of_slices,
            ]
            tiff_data = dataset.read(tiff_bbox["topleft"], padded_tiff_bbox_size)
        for slice_index in range(number_of_slices):
            slice_name_number = batch_number * batch_size + slice_index + 1
            if tiling_size is None:
                tiff_file_name = wkw_name_and_bbox_to_tiff_name(name, slice_name_number)
                tiff_file_path = dest_path / tiff_file_name

                image = wkw_slice_to_image(tiff_data[:, :, :, slice_index], downsample)
                image.save(tiff_file_path)
                logging.info("Saved slice %s", slice_name_number)

            else:
                for y_tile_index in range(ceil(tiff_bbox["size"][1] / tiling_size[1])):
                    tile_tiff_path = (
                        dest_path / str(slice_name_number) / str(y_tile_index + 1)
                    )
                    tile_tiff_path.mkdir(parents=True, exist_ok=True)
                    for x_tile_index in range(
                        ceil(tiff_bbox["size"][0] / tiling_size[0])
                    ):
                        tile_tiff_filename = f"{x_tile_index + 1}.tiff"
                        tile_image = wkw_slice_to_image(
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

                logging.info(f"saved tiles for slice {slice_name_number}")

    logging.info(f"saved all tiles of batch {batch_number}")


def export_tiff_stack(
    wkw_file_path: Path,
    wkw_layer: str,
    bbox: Dict[str, List[int]],
    mag: Mag,
    destination_path: Path,
    name: str,
    tiling_slice_size: Union[None, Tuple[int, int]],
    batch_size: int,
    downsample: int,
    args: Namespace,
) -> None:
    destination_path.mkdir(parents=True, exist_ok=True)
    dataset_path = wkw_file_path / wkw_layer / mag.to_layer_name()

    with get_executor_for_args(args) as executor:
        num_slices = ceil(bbox["size"][2] / batch_size)
        slices = range(0, num_slices)

        logging.info("starting jobs")
        futures = executor.map_to_futures(
            partial(
                export_tiff_slice,
                bbox,
                destination_path,
                name,
                dataset_path,
                tiling_slice_size,
                batch_size,
                downsample,
                mag,
            ),
            slices,
        )
        wait_and_ensure_success(futures)


def export_wkw_as_tiff(args: Namespace) -> None:
    setup_logging(args)

    if args.bbox is None:
        _, _, bbox_dim, origin = read_metadata_for_layer(
            args.source_path, args.layer_name
        )
        bbox = {"topleft": origin, "size": bbox_dim}
    else:
        bbox = {"topleft": list(args.bbox.topleft), "size": list(args.bbox.size)}

    logging.info(f"Starting tiff export for bounding box: {bbox}")

    if args.tiles_per_dimension is not None:
        args.tile_size = [int(s.strip()) for s in args.tiles_per_dimension.split(",")]
        assert len(args.tile_size) == 2
        logging.info(
            f"Using tiling with {args.tile_size[0]},{args.tile_size[1]} tiles in the dimensions."
        )
        args.tile_size[0] = ceil(bbox["size"][0] / args.tile_size[0])
        args.tile_size[1] = ceil(bbox["size"][1] / args.tile_size[1])

    elif args.tile_size is not None:
        args.tile_size = [int(s.strip()) for s in args.tile_size.split(",")]
        assert len(args.tile_size) == 2
        logging.info(
            f"Using tiling with the size of {args.tile_size[0]},{args.tile_size[1]}."
        )
    args.batch_size = int(args.batch_size)

    export_tiff_stack(
        wkw_file_path=args.source_path,
        wkw_layer=args.layer_name,
        bbox=bbox,
        mag=Mag(args.mag),
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
    arguments = create_parser().parse_args()
    export_wkw_as_tiff(arguments)
