import argparse
import logging
import time
from functools import partial
from pathlib import Path
from typing import Optional, Tuple, Union, cast

from cluster_tools import Executor
import numpy as np
import zarr
from webknossos import (
    BoundingBox,
    DataFormat,
    Dataset,
    Mag,
    MagView,
    SegmentationLayer,
    Vec3Int,
)
from webknossos.dataset._array import _fsstore_from_path
from webknossos.utils import get_executor_for_args, wait_and_ensure_success

from ._internal.utils import (
    add_data_format_flags,
    add_distribution_flags,
    add_interpolation_flag,
    add_sampling_mode_flag,
    add_verbose_flag,
    add_voxel_size_flag,
    parse_path,
    setup_logging,
    setup_warnings,
)

logger = logging.getLogger(__name__)


def parse_flip_axes(flip_axes: str) -> Tuple[int, ...]:
    try:
        indices = tuple(int(x) for x in flip_axes.split(","))
    except Exception as e:
        raise argparse.ArgumentTypeError("The flip_axes could not be parsed") from e
    if [i for i in indices if i < 0 or i > 3]:
        raise argparse.ArgumentTypeError("The flip_axes contains out-of-bound values")
    return indices


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "source_path",
        help="Path to zarr file to convert",
        type=parse_path,
    )

    parser.add_argument(
        "target_path",
        help="Output directory for the generated WKW dataset.",
        type=parse_path,
    )

    add_voxel_size_flag(parser, required=False)

    parser.add_argument(
        "--layer_name",
        "-l",
        help="Name of the cubed layer (color or segmentation).",
        default="color",
    )

    parser.add_argument(
        "--is_segmentation_layer",
        "-sl",
        help="Set whether this is a segmentation layer, defaulting to a color layer if nothing is set.",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--flip_axes",
        help="The axes which should be flipped. "
        "Input format is a comma separated list of axis indices. "
        "For example, 1,2,3 will flip the x, y and z axes.",
        default=None,
        type=parse_flip_axes,
    )

    parser.add_argument(
        "--no_compress",
        help="Don't compress saved data",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--max_mag",
        "-m",
        help="Max resolution to be downsampled. Needs to be a power of 2. In case of anisotropic downsampling, "
        "the process is considered done when max(current_mag) >= max(max_mag) where max takes the "
        "largest dimension of the mag tuple x, y, z. For example, a maximum mag value of 8 (or 8-8-8) "
        "will stop the downsampling as soon as a magnification is produced for which one dimension is "
        "equal or larger than 8. "
        "The default value is calculated depending on the dataset size. In the lowest Mag, the size will be "
        "smaller than 100vx per dimension",
        type=int,
        default=None,
    )

    add_sampling_mode_flag(parser)
    add_interpolation_flag(parser)
    add_data_format_flags(parser)

    add_distribution_flags(parser)

    add_verbose_flag(parser)

    return parser


def _zarr_chunk_converter(
    bounding_box: BoundingBox,
    source_zarr_path: Path,
    target_mag_view: MagView,
    flip_axes: Optional[Union[int, Tuple[int, ...]]],
) -> int:
    logging.info(f"Conversion of {bounding_box.topleft}")

    slices = bounding_box.to_slices()
    zarr_file = zarr.open(store=_fsstore_from_path(source_zarr_path), mode="r")
    source_data = zarr_file[slices][None, ...]

    if flip_axes:
        source_data = np.flip(source_data, flip_axes)

    contiguous_chunk = source_data.copy(order="F")
    target_mag_view.write(contiguous_chunk, bounding_box.topleft)

    return source_data.max()


def convert_zarr(
    source_zarr_path: Path,
    target_path: Path,
    layer_name: str,
    data_format: DataFormat,
    chunk_shape: Vec3Int,
    chunks_per_shard: Vec3Int,
    is_segmentation_layer: bool = False,
    voxel_size: Optional[Tuple[float, float, float]] = (1.0, 1.0, 1.0),
    flip_axes: Optional[Union[int, Tuple[int, ...]]] = None,
    compress: bool = True,
    executor: Optional[Executor] = None,
) -> MagView:
    ref_time = time.time()

    f = zarr.open(store=_fsstore_from_path(source_zarr_path), mode="r")
    input_dtype = f.dtype
    shape = f.shape

    if voxel_size is None:
        voxel_size = 1.0, 1.0, 1.0
    wk_ds = Dataset(target_path, voxel_size=voxel_size, exist_ok=True)
    wk_layer = wk_ds.get_or_add_layer(
        layer_name,
        "segmentation" if is_segmentation_layer else "color",
        dtype_per_layer=np.dtype(input_dtype),
        num_channels=1,
        largest_segment_id=0,
        data_format=data_format,
    )
    wk_layer.bounding_box = BoundingBox((0, 0, 0), shape)
    wk_mag = wk_layer.get_or_add_mag(
        "1",
        chunk_shape=chunk_shape,
        chunks_per_shard=chunks_per_shard,
        compress=compress,
    )

    # Parallel chunk conversion
    with get_executor_for_args(None, executor) as executor:
        largest_segment_id_per_chunk = wait_and_ensure_success(
            executor.map_to_futures(
                partial(
                    _zarr_chunk_converter,
                    source_zarr_path=source_zarr_path,
                    target_mag_view=wk_mag,
                    flip_axes=flip_axes,
                ),
                wk_layer.bounding_box.chunk(chunk_shape=chunk_shape * chunks_per_shard),
            )
        )

    if is_segmentation_layer:
        largest_segment_id = int(max(largest_segment_id_per_chunk))
        cast(SegmentationLayer, wk_layer).largest_segment_id = largest_segment_id

    logger.debug(
        "Conversion of {} took {:.8f}s".format(source_zarr_path, time.time() - ref_time)
    )
    return wk_mag


def main(args: argparse.Namespace) -> None:
    source_path = args.source_path

    if not source_path.is_dir():
        logger.error("source_path is not a directory")
        return

    with get_executor_for_args(args) as executor:
        mag_view = convert_zarr(
            source_path,
            args.target_path,
            layer_name=args.layer_name,
            data_format=args.data_format,
            chunk_shape=args.chunk_shape,
            chunks_per_shard=args.chunks_per_shard,
            is_segmentation_layer=args.is_segmentation_layer,
            voxel_size=args.voxel_size,
            flip_axes=args.flip_axes,
            compress=not args.no_compress,
            executor=executor,
        )

        mag_view.layer.downsample(
            from_mag=mag_view.mag,
            coarsest_mag=None if args.max_mag is None else Mag(args.max_mag),
            interpolation_mode=args.interpolation_mode,
            compress=not args.no_compress,
            sampling_mode=args.sampling_mode,
            executor=executor,
        )


if __name__ == "__main__":
    setup_warnings()
    args = create_parser().parse_args()
    setup_logging(args)

    main(args)
