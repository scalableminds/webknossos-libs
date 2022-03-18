from argparse import ArgumentParser, Namespace
from pathlib import Path

from typing import Optional
from webknossos.geometry import Vec3Int

from wkcuber.api.dataset import Dataset
from wkcuber.downsampling_utils import SamplingModes
from .mag import Mag

from .utils import (
    add_verbose_flag,
    add_distribution_flags,
    add_interpolation_flag,
    add_isotropic_flag,
    setup_logging,
    add_sampling_mode_flag,
)


def create_parser() -> ArgumentParser:
    parser = ArgumentParser()

    parser.add_argument("path", help="Directory containing the dataset.", type=Path)

    parser.add_argument(
        "--layer_name",
        "-l",
        help="Name of the cubed layer (color or segmentation)",
        default="color",
    )

    parser.add_argument(
        "--from_mag",
        "--from",
        "-f",
        help="Resolution to base upsampling on",
        type=str,
    )

    # Either provide the maximum resolution to be upsampled OR a specific, anisotropic magnification.
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--target_mag",
        "-t",
        help="Min resolution to be upsampled. In case of anisotropic upsampling, the process is considered "
        "done when max(current_mag) < max(target_mag) where max takes the largest dimension of the mag tuple "
        "x, y, z (or after Mag(1) was upsampled). For example, a minimum mag value of 2 (or 2-2-2) will stop "
        "the upsampling as soon as a magnification is produced for which one dimension is smaller than 2. "
        "The default value is Mag(1)",
        type=Mag,
        default=Mag(1),
    )

    group.add_argument(
        "--anisotropic_target_mag",
        help="Specify an explicit anisotropic target magnification (e.g., --anisotropic_target_mag 2-2-1)."
        "All magnifications until this target magnification will be created. Consider using --anisotropic "
        "instead which automatically creates multiple anisotropic magnifications depending "
        "on the dataset's scale",
        type=str,
    )

    parser.add_argument(
        "--buffer_cube_size",
        "-b",
        help="Size of buffered cube to be upsampled (i.e. buffer cube edge length)",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--no_compress",
        help="Don't compress data during upsampling",
        default=False,
        action="store_true",
    )

    add_interpolation_flag(parser)
    add_verbose_flag(parser)
    add_isotropic_flag(parser)
    add_sampling_mode_flag(parser)
    add_distribution_flags(parser)

    return parser


def upsample_mags(
    path: Path,
    layer_name: Optional[str] = None,
    from_mag: Optional[Mag] = None,
    target_mag: Mag = Mag(1),
    buffer_shape: Optional[Vec3Int] = None,
    compress: bool = True,
    args: Optional[Namespace] = None,
    sampling_mode: str = SamplingModes.ANISOTROPIC,
) -> None:
    assert layer_name and from_mag or not layer_name and not from_mag, (
        "You provided only one of the following "
        "parameters: layer_name, from_mag but both "
        "need to be set or none. If you don't provide "
        "the parameters you need to provide the path "
        "argument with the mag and layer to upsample"
        " (e.g dataset/color/1)."
    )
    if not layer_name or not from_mag:
        layer_name = path.parent.name
        from_mag = Mag(path.name)
        path = path.parent.parent

    Dataset.open(path).get_layer(layer_name).upsample(
        from_mag=from_mag,
        min_mag=target_mag,
        compress=compress,
        sampling_mode=sampling_mode,
        buffer_shape=buffer_shape,
        args=args,
    )


if __name__ == "__main__":
    args = create_parser().parse_args()
    setup_logging(args)

    if args.isotropic is not None:
        raise DeprecationWarning(
            "The flag 'isotropic' is deprecated. Consider using '--sampling_mode isotropic' instead."
        )

    if args.anisotropic_target_mag is not None:
        raise DeprecationWarning(
            "The 'anisotropic_target_mag' flag is deprecated. Use '--max' instead (and consider changing the 'sampling_mode')"
        )

    from_mag = Mag(args.from_mag)
    target_mag = Mag(args.target_mag)
    buffer_shape = (
        Vec3Int.full(args.buffer_cube_size)
        if args.buffer_cube_size is not None
        else None
    )

    upsample_mags(
        args.path,
        args.layer_name,
        from_mag,
        target_mag,
        buffer_shape=buffer_shape,
        compress=not args.no_compress,
        sampling_mode=args.sampling_mode,
        args=args,
    )
