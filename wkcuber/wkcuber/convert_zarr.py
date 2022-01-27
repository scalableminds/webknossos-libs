import argparse
from functools import partial
import logging
from pathlib import Path
import time
from typing import Optional, Tuple, Union, cast
import numpy as np
import zarr

from webknossos.dataset import SegmentationLayer

from webknossos.dataset.defaults import DEFAULT_WKW_FILE_LEN
from webknossos import BoundingBox, Dataset, Mag, MagView
from webknossos.utils import get_executor_for_args, wait_and_ensure_success
from .utils import (
    add_distribution_flags,
    add_interpolation_flag,
    add_sampling_mode_flag,
    add_scale_flag,
    add_verbose_flag,
    DEFAULT_WKW_VOXELS_PER_BLOCK,
    get_executor_args,
    setup_logging,
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
        type=Path,
    )

    parser.add_argument(
        "target_path", help="Output directory for the generated WKW dataset.", type=Path
    )

    add_scale_flag(parser, required=False)

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
    zarr_file = zarr.open(source_zarr_path, "r")
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
    is_segmentation_layer: bool = False,
    scale: Optional[Tuple[float, float, float]] = (1.0, 1.0, 1.0),
    flip_axes: Optional[Union[int, Tuple[int, ...]]] = None,
    compress: bool = True,
    file_len: int = DEFAULT_WKW_FILE_LEN,
    executor_args: Optional[argparse.Namespace] = None,
) -> MagView:
    ref_time = time.time()

    f = zarr.open(source_zarr_path, "r")
    input_dtype = f.dtype
    shape = f.shape

    if scale is None:
        scale = 1.0, 1.0, 1.0
    wk_ds = Dataset(target_path, scale=scale, exist_ok=True)
    wk_layer = wk_ds.get_or_add_layer(
        layer_name,
        "segmentation" if is_segmentation_layer else "color",
        dtype_per_layer=np.dtype(input_dtype),
        num_channels=1,
        largest_segment_id=0,
    )
    wk_layer.bounding_box = BoundingBox((0, 0, 0), shape)
    wk_mag = wk_layer.get_or_add_mag("1", file_len=file_len, compress=compress)

    # Parallel chunk conversion
    with get_executor_for_args(executor_args) as executor:
        largest_segment_id_per_chunk = wait_and_ensure_success(
            executor.map_to_futures(
                partial(
                    _zarr_chunk_converter,
                    source_zarr_path=source_zarr_path,
                    target_mag_view=wk_mag,
                    flip_axes=flip_axes,
                ),
                wk_layer.bounding_box.chunk(
                    chunk_size=(DEFAULT_WKW_VOXELS_PER_BLOCK * file_len,) * 3
                ),
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

    executor_args = get_executor_args(args)

    mag_view = convert_zarr(
        source_path,
        args.target_path,
        layer_name=args.layer_name,
        is_segmentation_layer=args.is_segmentation_layer,
        scale=args.scale,
        flip_axes=args.flip_axes,
        compress=not args.no_compress,
        executor_args=executor_args,
    )

    mag_view.layer.downsample(
        from_mag=mag_view.mag,
        max_mag=None if args.max_mag is None else Mag(args.max_mag),
        interpolation_mode=args.interpolation_mode,
        compress=not args.no_compress,
        sampling_mode=args.sampling_mode,
        args=executor_args,
    )


if __name__ == "__main__":
    args = create_parser().parse_args()
    setup_logging(args)

    main(args)
