from argparse import ArgumentParser
from os import path
from uuid import uuid4
import logging
import shutil

from .cubing import cubing, BLOCK_LEN
from .downsampling import downsample_mag, DEFAULT_EDGE_LEN
from .compress import compress_mag
from .metadata import write_webknossos_metadata
from .utils import add_verbose_flag, add_jobs_flag


def create_parser():
    parser = ArgumentParser()

    parser.add_argument("source_path", help="Directory containing the input images.")

    parser.add_argument(
        "target_path", help="Output directory for the generated dataset."
    )

    parser.add_argument(
        "--layer_name",
        "-l",
        help="Name of the cubed layer (color or segmentation)",
        default="color",
    )

    parser.add_argument(
        "--dtype",
        "-d",
        help="Target datatype (e.g. uint8, uint16, uint32)",
        default="uint8",
    )

    parser.add_argument(
        "--batch_size",
        "-b",
        help="Number of slices to buffer per job",
        default=BLOCK_LEN,
    )

    parser.add_argument(
        "--max_mag",
        "-m",
        help="Max resolution to be downsampled. Needs to be a power of 2.",
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

    add_verbose_flag(parser)
    add_jobs_flag(parser)

    return parser


def compress_mag_inplace(target_path, layer_name, mag, jobs):
    compress_target_path = "{}.compress-{}".format(target_path, uuid4())
    compress_mag(target_path, layer_name, compress_target_path, mag, jobs)

    shutil.rmtree(path.join(args.target_path, args.layer_name, str(mag)))
    shutil.move(
        path.join(compress_target_path, layer_name, str(mag)),
        path.join(target_path, layer_name, str(mag)),
    )
    shutil.rmtree(compress_target_path)


if __name__ == "__main__":
    args = create_parser().parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    cubing(
        args.source_path,
        args.target_path,
        args.layer_name,
        args.dtype,
        int(args.batch_size),
        int(args.jobs),
    )

    if not args.no_compress:
        compress_mag_inplace(args.target_path, args.layer_name, 1, int(args.jobs))

    target_mag = 2
    while target_mag <= int(args.max_mag):
        source_mag = target_mag // 2
        downsample_mag(
            args.target_path,
            args.layer_name,
            source_mag,
            target_mag,
            args.dtype,
            "default",
            DEFAULT_EDGE_LEN,
            int(args.jobs),
        )
        if not args.no_compress:
            compress_mag_inplace(
                args.target_path, args.layer_name, target_mag, int(args.jobs)
            )
        target_mag = target_mag * 2

    scale = tuple(float(x) for x in args.scale.split(","))
    write_webknossos_metadata(args.target_path, args.name, scale, compute_max_id=False)
