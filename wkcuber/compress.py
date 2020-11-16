import time
import wkw
import shutil
import logging
from argparse import ArgumentParser, Namespace
from os import path, makedirs
from uuid import uuid4
from .mag import Mag

from .utils import (
    add_verbose_flag,
    open_wkw,
    WkwDatasetInfo,
    add_distribution_flags,
    get_executor_for_args,
    wait_and_ensure_success,
    setup_logging,
)
from .metadata import detect_resolutions, convert_element_class_to_dtype
from typing import List, Tuple

BACKUP_EXT = ".bak"


def create_parser() -> ArgumentParser:
    parser = ArgumentParser()

    parser.add_argument(
        "source_path", help="Directory containing the source WKW dataset."
    )

    parser.add_argument(
        "target_path",
        help="Output directory for the compressed WKW dataset.",
        nargs="?",
        default=None,
    )

    parser.add_argument(
        "--layer_name",
        "-l",
        help="Name of the cubed layer (color or segmentation)",
        default="color",
    )

    parser.add_argument(
        "--mag", "-m", nargs="*", help="Magnification level", default=None
    )

    add_verbose_flag(parser)
    add_distribution_flags(parser)

    return parser


def compress_file_job(args: Tuple[str, str]) -> None:
    source_path, target_path = args
    try:
        logging.debug("Compressing '{}' to '{}'".format(source_path, target_path))
        ref_time = time.time()

        makedirs(path.dirname(target_path), exist_ok=True)
        wkw.File.compress(source_path, target_path)

        if not path.exists(target_path):
            raise Exception("Did not create compressed file {}".format(target_path))

        logging.debug(
            "Compressing of '{}' took {:.8f}s".format(
                source_path, time.time() - ref_time
            )
        )
    except Exception as exc:
        logging.error("Compressing of '{}' failed with {}".format(source_path, exc))
        raise exc


def compress_mag(
    source_path: str,
    layer_name: str,
    target_path: str,
    mag: Mag,
    args: Namespace = None,
) -> None:
    if path.exists(path.join(target_path, layer_name, str(mag))):
        logging.error("Target path '{}' already exists".format(target_path))
        exit(1)

    if args is not None and hasattr(args, "dtype"):
        header = wkw.Header(convert_element_class_to_dtype(args.dtype))
    else:
        header = None
    source_wkw_info = WkwDatasetInfo(source_path, layer_name, mag, header)
    target_mag_path = path.join(target_path, layer_name, str(mag))
    logging.info("Compressing mag {0} in '{1}'".format(str(mag), target_mag_path))

    with open_wkw(source_wkw_info) as source_wkw:
        source_wkw.compress(target_mag_path)
        with get_executor_for_args(args) as executor:
            job_args = []
            for file in source_wkw.list_files():
                rel_file = path.relpath(file, source_wkw.root)
                job_args.append((file, path.join(target_mag_path, rel_file)))

            wait_and_ensure_success(
                executor.map_to_futures(compress_file_job, job_args)
            )

    logging.info("Mag {0} successfully compressed".format(str(mag)))


def compress_mag_inplace(
    target_path: str, layer_name: str, mag: Mag, args: Namespace = None
) -> None:
    compress_target_path = "{}.compress-{}".format(target_path, uuid4())
    compress_mag(target_path, layer_name, compress_target_path, mag, args)

    shutil.rmtree(path.join(target_path, layer_name, str(mag)))
    shutil.move(
        path.join(compress_target_path, layer_name, str(mag)),
        path.join(target_path, layer_name, str(mag)),
    )
    shutil.rmtree(compress_target_path)


def compress_mags(
    source_path: str,
    layer_name: str,
    target_path: str = None,
    mags: List[Mag] = None,
    args: Namespace = None,
) -> None:
    if target_path is None:
        target = source_path + ".tmp"
    else:
        target = target_path

    if mags is None:
        mags = list(detect_resolutions(source_path, layer_name))
    mags.sort()

    for mag in mags:
        compress_mag(source_path, layer_name, target, mag, args)

    if target_path is None:
        makedirs(path.join(source_path + BACKUP_EXT, layer_name), exist_ok=True)
        for mag in mags:
            shutil.move(
                path.join(source_path, layer_name, str(mag)),
                path.join(source_path + BACKUP_EXT, layer_name, str(mag)),
            )
            shutil.move(
                path.join(target, layer_name, str(mag)),
                path.join(source_path, layer_name, str(mag)),
            )
        shutil.rmtree(target)
        logging.info(
            "Old files are still present in '{0}.bak'. Please remove them when not required anymore.".format(
                source_path
            )
        )


if __name__ == "__main__":
    args = create_parser().parse_args()
    setup_logging(args)
    compress_mags(args.source_path, args.layer_name, args.target_path, args.mag, args)
