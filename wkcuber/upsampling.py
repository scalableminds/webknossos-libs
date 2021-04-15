from argparse import ArgumentParser, Namespace
import os

from wkcuber.api.Dataset import WKDataset
from .mag import Mag

from .utils import (
    add_verbose_flag,
    add_distribution_flags,
    add_interpolation_flag,
    add_isotropic_flag,
    setup_logging,
)


def create_parser() -> ArgumentParser:
    parser = ArgumentParser()

    parser.add_argument("path", help="Directory containing the dataset.")

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
        type=int,
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
    add_distribution_flags(parser)

    return parser


def upsample_mags(
    path: str,
    layer_name: str = None,
    from_mag: Mag = None,
    target_mag: Mag = Mag(1),
    buffer_edge_len: int = None,
    compress: bool = True,
    args: Namespace = None,
    anisotropic: bool = True,
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
        layer_name = os.path.basename(os.path.dirname(path))
        from_mag = Mag(os.path.basename(path))
        path = os.path.dirname(os.path.dirname(path))

    WKDataset(path).get_layer(layer_name).upsample(
        from_mag=from_mag,
        min_mag=target_mag,
        compress=compress,
        anisotropic=anisotropic,
        buffer_edge_len=buffer_edge_len,
        args=args,
    )


if __name__ == "__main__":
    args = create_parser().parse_args()
    setup_logging(args)

    from_mag = Mag(args.from_mag)
    target_mag = (
        Mag(args.anisotropic_target_mag)
        if args.anisotropic_target_mag
        else Mag(args.target_mag)
    )

    upsample_mags(
        args.path,
        args.layer_name,
        from_mag,
        target_mag,
        buffer_edge_len=args.buffer_cube_size,
        compress=not args.no_compress,
        anisotropic=not args.isotropic,
        args=args,
    )
