from pathlib import Path

from typing import Optional

from argparse import ArgumentParser, Namespace
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
    get_executor_args,
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
        help="Resolution to base downsampling on",
        type=str,
        default="1",
    )

    # Either provide the maximum resolution to be upsampled OR a specific, anisotropic magnification.
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--max",
        "-m",
        help="Max resolution to be downsampled. Needs to be a power of 2. In case of anisotropic downsampling, "
        "the process is considered done when max(current_mag) >= max(max_mag) where max takes the "
        "largest dimension of the mag tuple x, y, z. For example, a maximum mag value of 8 (or 8-8-8) "
        "will stop the downsampling as soon as a magnification is produced for which one dimension is "
        "equal or larger than 8. "
        "The default value is calculated depending on the dataset size. In the lowest Mag, the size will be "
        "smaller than 100vx per dimension",
        type=str,
        default=None,
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
        help="Size of buffered cube to be downsampled (i.e. buffer cube edge length)",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--no_compress",
        help="Don't compress data during downsampling",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--force_sampling_scheme",
        default=False,
        type=bool,
        help="Specifies if the layer should even be downsampled if the calculated downsampling scheme is not supported by webknossos. Depending on this flag, a warning or an error is raised.",
    )

    add_interpolation_flag(parser)
    add_verbose_flag(parser)
    add_sampling_mode_flag(parser)
    add_isotropic_flag(parser)
    add_distribution_flags(parser)

    return parser


def downsample_mags(
    path: Path,
    layer_name: Optional[str] = None,
    from_mag: Optional[Mag] = None,
    max_mag: Optional[Mag] = None,
    interpolation_mode: str = "default",
    buffer_shape: Optional[Vec3Int] = None,
    compress: bool = True,
    args: Optional[Namespace] = None,
    sampling_mode: str = SamplingModes.ANISOTROPIC,
    force_sampling_scheme: bool = False,
) -> None:
    """
    Argument `path` expects the directory containing the dataset.
    Argument `layer_name` expects the name of the layer (color or segmentation).
    Argument `from_mag` expects the resolution to base downsampling on.

    For the other parameters see the CLI help or `Layer.downsample` and `Layer.downsampling_mag`.
    """
    assert layer_name and from_mag or not layer_name and not from_mag, (
        "You provided only one of the following "
        "parameters: layer_name, from_mag but both "
        "need to be set or none. If you don't provide "
        "the parameters you need to provide the path "
        "argument with the mag and layer to downsample"
        " (e.g dataset/color/1)."
    )
    if not layer_name or not from_mag:
        layer_name = path.parent.name
        from_mag = Mag(path.name)
        path = path.parent.parent

    Dataset.open(path).get_layer(layer_name).downsample(
        from_mag=from_mag,
        max_mag=max_mag,
        interpolation_mode=interpolation_mode,
        compress=compress,
        sampling_mode=sampling_mode,
        buffer_shape=buffer_shape,
        force_sampling_scheme=force_sampling_scheme,
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
    max_mag = None if args.max is None else Mag(args.max)
    buffer_shape = (
        Vec3Int.full(args.buffer_cube_size)
        if args.buffer_cube_size is not None
        else None
    )

    downsample_mags(
        path=args.path,
        layer_name=args.layer_name,
        from_mag=from_mag,
        max_mag=max_mag,
        interpolation_mode=args.interpolation_mode,
        compress=not args.no_compress,
        sampling_mode=args.sampling_mode,
        buffer_shape=buffer_shape,
        force_sampling_scheme=args.force_sampling_scheme,
        args=get_executor_args(args),
    )
