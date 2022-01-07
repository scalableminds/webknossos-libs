import argparse
from functools import partial
import logging
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np

from webknossos.dataset.defaults import DEFAULT_WKW_FILE_LEN
from webknossos import BoundingBox, Dataset, Mag, MagView
from webknossos.utils import (
    get_executor_for_args,
    time_start,
    time_stop,
    wait_and_ensure_success,
)
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


def parse_shape(shape: str) -> Tuple[float, ...]:
    try:
        return tuple(int(x) for x in shape.split(","))
    except Exception as e:
        raise argparse.ArgumentTypeError("The shape could not be parsed") from e


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
        help="Path to raw file to convert",
        type=Path,
    )

    parser.add_argument(
        "target_path", help="Output directory for the generated WKW dataset.", type=Path
    )

    parser.add_argument(
        "--input_dtype",
        help="Input dataset datatype (e.g. uint8, uint16, float32).",
        required=True,
    )

    parser.add_argument(
        "--shape",
        help="Shape of the dataset (width, height, depth)",
        type=parse_shape,
        required=True,
    )

    add_scale_flag(parser, required=False)

    parser.add_argument(
        "--layer_name",
        "-l",
        help="Name of the cubed layer (color or segmentation).",
        default="color",
    )

    parser.add_argument(
        "--order",
        help="The input data storage layout:"
        "either 'F' for Fortran-style/column-major order (the default), "
        "or 'C' for C-style/row-major order. "
        "Note: Axes are expected in  (x, y, z) order.",
        choices=("C", "F"),
        default="F",
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


def _raw_chunk_converter(
    bounding_box: BoundingBox,
    source_raw_path: Path,
    target_mag_view: MagView,
    input_dtype: str,
    shape: Tuple[int, int, int],
    order: str,
    flip_axes: Optional[Union[int, Tuple[int, ...]]],
) -> None:
    logging.info(f"Conversion of {bounding_box.topleft}")
    source_data = np.memmap(
        source_raw_path, dtype=input_dtype, mode="r", shape=(1,) + shape, order=order
    )

    if flip_axes:
        source_data = np.flip(source_data, flip_axes)

    contiguous_chunk = source_data[(slice(None),) + bounding_box.to_slices()].copy(
        order="F"
    )
    target_mag_view.write(contiguous_chunk, bounding_box.topleft)


def convert_raw(
    source_raw_path: Path,
    target_path: Path,
    layer_name: str,
    input_dtype: str,
    shape: Tuple[int, int, int],
    order: str = "F",
    scale: Optional[Tuple[float, float, float]] = (1.0, 1.0, 1.0),
    flip_axes: Optional[Union[int, Tuple[int, ...]]] = None,
    compress: bool = True,
    file_len: int = DEFAULT_WKW_FILE_LEN,
    executor_args: Optional[argparse.Namespace] = None,
) -> MagView:
    assert order in ("C", "F")
    time_start(f"Conversion of {source_raw_path}")

    if scale is None:
        scale = 1.0, 1.0, 1.0
    wk_ds = Dataset(target_path, scale=scale, exist_ok=True)
    wk_layer = wk_ds.get_or_add_layer(
        layer_name,
        "color",
        dtype_per_layer=np.dtype(input_dtype),
        num_channels=1,
    )
    wk_layer.bounding_box = BoundingBox((0, 0, 0), shape)
    wk_mag = wk_layer.get_or_add_mag("1", file_len=file_len, compress=compress)

    # Parallel chunk conversion
    with get_executor_for_args(executor_args) as executor:
        wait_and_ensure_success(
            executor.map_to_futures(
                partial(
                    _raw_chunk_converter,
                    source_raw_path=source_raw_path,
                    target_mag_view=wk_mag,
                    input_dtype=input_dtype,
                    shape=shape,
                    order=order,
                    flip_axes=flip_axes,
                ),
                wk_layer.bounding_box.chunk(
                    chunk_size=(DEFAULT_WKW_VOXELS_PER_BLOCK * file_len,) * 3
                ),
            )
        )

    time_stop(f"Conversion of {source_raw_path}")
    return wk_mag


def main(args: argparse.Namespace) -> None:
    source_path = args.source_path

    if source_path.is_dir():
        logger.error("source_path is not a file")
        return

    executor_args = get_executor_args(args)

    mag_view = convert_raw(
        source_path,
        args.target_path,
        args.layer_name,
        args.input_dtype,
        args.shape,
        args.order,
        args.scale,
        args.flip_axes,
        not args.no_compress,
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
