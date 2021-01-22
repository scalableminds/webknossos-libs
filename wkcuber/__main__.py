from .cubing import cubing, add_cubing_arguments
from .downsampling import downsample_mags_isotropic, downsample_mags_anisotropic
from .compress import compress_mag_inplace
from .metadata import write_webknossos_metadata, refresh_metadata
from .utils import add_isotropic_flag, setup_logging, add_scale_flag, add_base_flags
from .mag import Mag
from argparse import Namespace, ArgumentParser
from os import path
from pathlib import Path
from .utils import find_files
from .convert_nifti import add_nifti_arguments, main as convert_nifti
from typing import List

SUPPORTED_FILE_TYPES = {
    ".wkw",
    ".nii",
    ".tif",
    ".tiff",
    ".jpg",
    ".jpeg",
    ".png",
    ".dm3",
    ".dm4",
}


def create_parser() -> ArgumentParser:
    parser = ArgumentParser()

    add_base_flags(parser)
    add_cubing_arguments(parser)
    add_nifti_arguments(parser)

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

    add_isotropic_flag(parser)

    return parser


def find_source_filenames(source_path: str) -> List[str]:
    # Find all source files that have a matching file extension
    input_path = Path(source_path)

    if input_path.is_dir():
        joined_path = path.join(source_path, "**")
    else:
        joined_path = source_path

    source_files = list(find_files(joined_path, SUPPORTED_FILE_TYPES))

    assert len(source_files) > 0, (
            "No image files found in path "
            + source_path
            + ". Supported suffixes are "
            + str(SUPPORTED_FILE_TYPES)
            + "."
    )

    _, ext = path.splitext(source_files[0])

    assert all(
        map(lambda p: path.splitext(p)[1] == ext, source_files)
    ), "Not all image files are of the same type"

    return source_files


def detect_conversion_type(args: Namespace):
    source_files = find_source_filenames(args.source_path)

    _, ext = path.splitext(source_files[0])

    if ext == ".wkw":
        print("Already WKW Dataset. Exiting...")
        exit(0)
    elif ext == ".nii":
        print("Converting Nifti Dataset")
        convert_nifti(args)
    else:
        main(args)


def main(args: Namespace) -> None:
    setup_logging(args)

    bounding_box = cubing(
        args.source_path,
        args.target_path,
        args.layer_name,
        args.dtype,
        args.batch_size,
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
    parsed_args: Namespace = create_parser().parse_args()
    detect_conversion_type(parsed_args)
