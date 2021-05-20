from wkcuber import downsample_mags
from .cubing import cubing, create_parser as create_cubing_parser
from .compress import compress_mag_inplace
from .metadata import write_webknossos_metadata, refresh_metadata
from .utils import (
    add_isotropic_flag,
    setup_logging,
    add_scale_flag,
    add_sampling_mode_flag,
    get_executor_args,
)
from .mag import Mag
from argparse import Namespace, ArgumentParser


def create_parser() -> ArgumentParser:
    parser = create_cubing_parser()

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

    arg_dict = vars(args)

    bounding_box = cubing(
        args.source_path,
        args.target_path,
        args.layer_name,
        arg_dict.get("batch_size"),
        arg_dict.get("channel_index"),
        arg_dict.get("dtype"),
        args.target_mag,
        args.wkw_file_len,
        args.interpolation_mode,
        args.start_z,
        args.pad,
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
        max_mag=None if args.max_mag is None else Mag(args.max_mag),
        interpolation_mode="default",
        compress=not args.no_compress,
        sampling_mode=args.sampling_mode,
        args=get_executor_args(args),
    )

    refresh_metadata(args.target_path)


if __name__ == "__main__":
    args = create_parser().parse_args()
    setup_logging(args)
    main(args)
