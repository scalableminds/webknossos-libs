from .cubing import cubing, create_parser as create_cubing_parser
from .downsampling import downsample_mags_isotropic, downsample_mags_anisotropic
from .compress import compress_mag_inplace
from .metadata import write_webknossos_metadata, refresh_metadata
from .utils import add_isotropic_flag, setup_logging, add_scale_flag
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

    return parser


def main(args: Namespace) -> None:
    setup_logging(args)

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

    if not args.isotropic:
        downsample_mags_anisotropic(
            args.target_path,
            args.layer_name,
            Mag(1),
            Mag(args.max_mag),
            args.scale,
            "default",
            not args.no_compress,
            args=args,
        )

    else:
        downsample_mags_isotropic(
            args.target_path,
            args.layer_name,
            Mag(1),
            Mag(args.max_mag),
            "default",
            not args.no_compress,
            args=args,
        )

    refresh_metadata(args.target_path)


if __name__ == "__main__":
    args = create_parser().parse_args()
    setup_logging(args)
    main(args)
