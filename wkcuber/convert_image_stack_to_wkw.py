from wkcuber import downsample_mags
from .cubing import cubing, create_parser as create_cubing_parser
from .compress import compress_mag_inplace
from .metadata import write_webknossos_metadata, refresh_metadata
from .utils import (
    add_isotropic_flag,
    setup_logging,
    add_scale_flag,
    add_sampling_mode_flag,
)
from .mag import Mag
from argparse import Namespace, ArgumentParser


def create_parser() -> ArgumentParser:
    parser = create_cubing_parser()

    parser.add_argument(
        "--max_mag",
        "-m",
        help="Max resolution to be downsampled. Needs to be a power of 2.",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--no_compress",
        help="Don't compress this data",
        default=False,
        action="store_true",
    )

    parser.add_argument("--name", "-n", help="Name of the dataset", default=None)
    add_scale_flag(parser)
    add_isotropic_flag(parser)
    add_sampling_mode_flag(parser)

    return parser


def main(args: Namespace) -> None:
    setup_logging(args)

    if args.isotropic is not None:
        raise DeprecationWarning(
            "The flag 'isotropic' is deprecated. Consider using '--sampling_mode isotropic' instead."
        )

    bounding_box = cubing(
        args.source_path,
        args.target_path,
        args.layer_name,
        args.batch_size if "batch_size" in args else None,
        args,
    )

    write_webknossos_metadata(
        args.target_path,
        args.name,
        args.scale,
        compute_max_id=False,
        exact_bounding_box=bounding_box,
    )

    if not args.no_compress:
        compress_mag_inplace(args.target_path, args.layer_name, Mag(1), args)

    downsample_mags(
        path=args.target_path,
        layer_name=args.layer_name,
        from_mag=Mag(1),
        max_mag=Mag(args.max_mag),
        interpolation_mode="default",
        compress=not args.no_compress,
        sampling_mode=args.sampling_mode,
        args=args,
    )

    refresh_metadata(args.target_path)


if __name__ == "__main__":
    args = create_parser().parse_args()
    setup_logging(args)
    main(args)
