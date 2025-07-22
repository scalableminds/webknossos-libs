"""This module takes care of exporting tiff images."""

import logging
from argparse import Namespace
from functools import partial
from math import ceil
from multiprocessing import cpu_count
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer
from PIL import Image
from scipy.ndimage import zoom

from ..dataset import Dataset, MagView, View
from ..dataset.defaults import DEFAULT_CHUNK_SHAPE
from ..geometry import BoundingBox, Mag, Vec3Int
from ..utils import get_executor_for_args, wait_and_ensure_success
from ._utils import (
    DistributionStrategy,
    Vec2Int,
    parse_bbox,
    parse_mag,
    parse_path,
    parse_vec2int,
)

logger = logging.getLogger(__name__)


def _make_tiff_name(name: str, slice_index: int) -> str:
    if name is None or name == "":
        return f"{slice_index:06d}.tiff"
    else:
        return f"{name}_{slice_index:06d}.tiff"


def _slice_to_image(data_slice: np.ndarray, downsample: int = 1) -> Image.Image:
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
    tiling_size: None | tuple[int, int],
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
            logger.debug("Saved slice %s", slice_name_number)

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
                            x_tile_index * tiling_size[0] : (x_tile_index + 1)
                            * tiling_size[0],
                            y_tile_index * tiling_size[1] : (y_tile_index + 1)
                            * tiling_size[1],
                            slice_index,
                        ],
                        downsample,
                    )

                    tile_image.save(tile_tiff_path / tile_tiff_filename)

            logger.debug("Saved tiles for slice %s", slice_name_number)

    logger.debug("Saved all tiles of bbox %s", tiff_bbox)


def export_tiff_stack(
    mag_view: MagView,
    bbox: BoundingBox,
    destination_path: Path,
    name: str,
    tiling_slice_size: None | tuple[int, int],
    batch_size: int,
    downsample: int,
    args: Namespace,
) -> None:
    destination_path.mkdir(parents=True, exist_ok=True)

    with get_executor_for_args(args) as executor:
        view = mag_view.get_view(
            absolute_offset=bbox.topleft, size=bbox.size, read_only=True
        )

        view_chunks = [
            view.get_view(
                relative_offset=(0, 0, z),
                size=(bbox.size.x, bbox.size.y, min(batch_size, bbox.size.z - z)),
                read_only=True,
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
            executor=executor,
            progress_desc="Exporting tiff files",
        )


def main(
    *,
    source: Annotated[
        Any,
        typer.Argument(
            help="Path to your raw image data.",
            show_default=False,
            parser=parse_path,
        ),
    ],
    target: Annotated[
        Any,
        typer.Argument(
            help="Target path to save your WEBKNOSSOS dataset.",
            show_default=False,
            parser=parse_path,
        ),
    ],
    layer_name: Annotated[
        str,
        typer.Option(help="Name of the cubed layer (color or segmentation)"),
    ] = "color",
    bbox: Annotated[
        BoundingBox | None,
        typer.Option(
            help="The BoundingBox that should be exported. "
            "If the input data is too small, it will be padded. If it's too large, it will be "
            "cropped. The input format is x,y,z,width,height,depth.",
            parser=parse_bbox,
            metavar="BBOX",
        ),
    ] = None,
    mag: Annotated[
        Mag,
        typer.Option(
            help="Max resolution to be downsampled. "
            "Should be number or minus separated string (e.g. 2 or 2-2-2).",
            parser=parse_mag,
            metavar="MAG",
        ),
    ] = 1,  # type: ignore
    name: Annotated[str, typer.Option(help="Name of the tiffs.")] = "",
    downsample: Annotated[int, typer.Option(help="Downsample each tiff image.")] = 1,
    tiles_per_dimension: Annotated[
        Vec2Int | None,
        typer.Option(
            help="For very large datasets, it is recommended to enable tiling which will ensure "
            "that each slice is exported to multiple images (i.e., tiles). As a parameter you "
            'should provide the amount of tiles per dimension in the form of "x,y".'
            'Also see at "--tile_size" to specify the absolute size of the tiles.',
            parser=parse_vec2int,
        ),
    ] = None,
    tile_size: Annotated[
        Vec2Int | None,
        typer.Option(
            help="For very large datasets, it is recommended to enable tiling which will ensure "
            "that each slice is exported to multiple images (i.e., tiles). As a parameter you "
            'should provide the the size of each tile per dimension in the form of "x,y".'
            'Also see at "--tiles_per_dimension" to specify the number of tiles in the dimensions.',
            parser=parse_vec2int,
        ),
    ] = None,
    batch_size: Annotated[
        int, typer.Option(help="Number of sections to buffer per job.")
    ] = DEFAULT_CHUNK_SHAPE.z,
    jobs: Annotated[
        int,
        typer.Option(
            help="Number of processes to be spawned.",
            rich_help_panel="Executor options",
        ),
    ] = cpu_count(),
    distribution_strategy: Annotated[
        DistributionStrategy,
        typer.Option(
            help="Strategy to distribute the task across CPUs or nodes.",
            rich_help_panel="Executor options",
        ),
    ] = DistributionStrategy.MULTIPROCESSING,
    job_resources: Annotated[
        str | None,
        typer.Option(
            help="Necessary when using slurm as distribution strategy. Should be a JSON string "
            '(e.g., --job-resources=\'{"mem": "10M"}\')\'',
            rich_help_panel="Executor options",
        ),
    ] = None,
) -> None:
    """Export your WEBKNOSSOS dataset to TIFF image data."""

    mag_view = Dataset.open(source).get_layer(layer_name).get_mag(mag)

    bbox = BoundingBox.from_ndbbox(mag_view.bounding_box) if bbox is None else bbox
    bbox = bbox.align_with_mag(mag_view.mag)

    logger.info("Starting tiff export for bounding box: %s", bbox)
    executor_args = Namespace(
        jobs=jobs,
        distribution_strategy=distribution_strategy.value,
        job_resources=job_resources,
    )
    used_tile_size = None
    if tiles_per_dimension is not None:
        tile_size = tiles_per_dimension
        logger.info(
            "Using tiling with %d,%d tiles in the dimensions.",
            tile_size[0],
            tile_size[1],
        )
        used_tile_size = (
            ceil(bbox.in_mag(mag_view.mag).size.x / tile_size[0]),
            ceil(bbox.in_mag(mag_view.mag).size.y / tile_size[1]),
        )

    elif tile_size is not None:
        logger.info("Using tiling with the size of %d,%d.", tile_size[0], tile_size[1])
        used_tile_size = tile_size

    export_tiff_stack(
        mag_view=mag_view,
        bbox=bbox,
        destination_path=target,
        name=name,
        tiling_slice_size=used_tile_size,
        batch_size=batch_size,
        downsample=downsample,
        args=executor_args,
    )
