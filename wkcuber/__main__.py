from argparse import ArgumentParser

import logging

from .cubing import cubing, BLOCK_LEN, create_parser as create_cubing_parser
from .downsampling import downsample_mags, downsample_mags_anisotropic, DEFAULT_EDGE_LEN
from .compress import compress_mag_inplace
from .metadata import write_webknossos_metadata
from .utils import add_verbose_flag, add_distribution_flags, add_anisotropic_flag
from .mag import Mag


def create_parser():
    parser = create_cubing_parser()

    parser.add_argument(
        "--max_mag",
        "-m",
        help="Max resolution to be downsampled. Needs to be a power of 2.",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--no_compress",
        help="Don't compress this data",
        default=False,
        action="store_true",
    )

    parser.add_argument("--name", "-n", help="Name of the dataset", default=None)

    parser.add_argument(
        "--scale",
        "-s",
        help="Scale of the dataset (e.g. 11.2,11.2,25). This is the size of one voxel in nm.",
        default="1,1,1",
    )

    add_anisotropic_flag(parser)

    return parser


def setup_logging(args):

    logging.basicConfig(
        level=(logging.DEBUG if args.verbose else logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )


if __name__ == "__main__":
    args = create_parser().parse_args()
    setup_logging(args)

    scale = tuple(float(x) for x in args.scale.split(","))
    bounding_box = cubing(
        args.source_path,
        args.target_path,
        args.layer_name,
        args.dtype,
        args.batch_size,
        args,
    )

    if not args.no_compress:
        compress_mag_inplace(args.target_path, args.layer_name, Mag(1), args)

    if args.anisotropic:
        downsample_mags_anisotropic(
            args.target_path,
            args.layer_name,
            Mag(1),
            Mag(args.max_mag),
            scale,
            "default",
            DEFAULT_EDGE_LEN,
            not args.no_compress,
            args,
        )

    else:
        downsample_mags(
            args.target_path,
            args.layer_name,
            Mag(1),
            Mag(args.max_mag),
            "default",
            DEFAULT_EDGE_LEN,
            not args.no_compress,
            args,
        )

    write_webknossos_metadata(
        args.target_path,
        args.name,
        scale,
        compute_max_id=False,
        exact_bounding_box=bounding_box,
    )
