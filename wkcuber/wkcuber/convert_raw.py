import argparse
import logging
from pathlib import Path
import time
from typing import Optional, Tuple, Union
import numpy as np

from webknossos.dataset.defaults import DEFAULT_WKW_FILE_LEN
from webknossos import Dataset, Mag, MagView
from .utils import (
    add_interpolation_flag,
    add_sampling_mode_flag,
    add_scale_flag,
    add_verbose_flag,
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

    add_verbose_flag(parser)

    return parser


def convert_raw(
    source_raw_path: Path,
    target_path: Path,
    layer_name: str,
    input_dtype: Optional[str],
    shape: Optional[Tuple[int, int, int]],
    order: str = "F",
    scale: Optional[Tuple[float, float, float]] = (1.0, 1.0, 1.0),
    flip_axes: Optional[Union[int, Tuple[int, ...]]] = None,
    compress: bool = True,
    file_len: int = DEFAULT_WKW_FILE_LEN,
) -> MagView:
    assert order in ("C", "F")
    ref_time = time.time()

    # Axes are understood as x,y,z ordered
    cube_data = np.memmap(
        source_raw_path, dtype=input_dtype, mode="r", shape=(1,) + shape, order=order
    )

    if flip_axes:
        cube_data = np.flip(cube_data, flip_axes)

    if scale is None:
        scale = 1.0, 1.0, 1.0
    wk_ds = Dataset.get_or_create(target_path, scale=scale)
    wk_layer = wk_ds.get_or_add_layer(
        layer_name,
        "color",
        dtype_per_layer=np.dtype(input_dtype),
        num_channels=1,
    )
    wk_mag = wk_layer.get_or_add_mag("1", file_len=file_len, compress=compress)
    wk_mag.write(cube_data)

    logger.debug(
        "Converting of {} took {:.8f}s".format(source_raw_path, time.time() - ref_time)
    )
    return wk_mag


def main(args: argparse.Namespace) -> None:
    source_path = args.source_path

    if source_path.is_dir():
        logger.error("source_path is not a file")
        return

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
    )

    mag_view.layer.downsample(
        from_mag=mag_view.mag,
        max_mag=None if args.max_mag is None else Mag(args.max_mag),
        interpolation_mode=args.interpolation_mode,
        compress=not args.no_compress,
        sampling_mode=args.sampling_mode,
    )


if __name__ == "__main__":
    args = create_parser().parse_args()
    setup_logging(args)

    main(args)
