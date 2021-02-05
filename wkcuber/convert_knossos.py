import time
import logging
from typing import Tuple, cast

import wkw
from argparse import ArgumentParser, Namespace

from .utils import (
    add_verbose_flag,
    open_wkw,
    open_knossos,
    WkwDatasetInfo,
    KnossosDatasetInfo,
    ensure_wkw,
    add_distribution_flags,
    get_executor_for_args,
    wait_and_ensure_success,
    setup_logging,
)
from .knossos import CUBE_EDGE_LEN
from .metadata import convert_element_class_to_dtype


def create_parser() -> ArgumentParser:
    parser = ArgumentParser()

    parser.add_argument(
        "source_path", help="Directory containing the source KNOSSOS dataset."
    )

    parser.add_argument(
        "target_path", help="Output directory for the generated WKW dataset."
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

    parser.add_argument("--mag", "-m", help="Magnification level", type=int, default=1)

    add_verbose_flag(parser)
    add_distribution_flags(parser)

    return parser


def convert_cube_job(
    args: Tuple[Tuple[int, int, int], KnossosDatasetInfo, WkwDatasetInfo]
) -> None:
    cube_xyz, source_knossos_info, target_wkw_info = args
    logging.info("Converting {},{},{}".format(cube_xyz[0], cube_xyz[1], cube_xyz[2]))
    ref_time = time.time()
    offset = cast(Tuple[int, int, int], tuple(x * CUBE_EDGE_LEN for x in cube_xyz))
    size = cast(Tuple[int, int, int], (CUBE_EDGE_LEN,) * 3)

    with open_knossos(source_knossos_info) as source_knossos, open_wkw(
        target_wkw_info
    ) as target_wkw:
        cube_data = source_knossos.read(offset, size)
        target_wkw.write(offset, cube_data)
    logging.debug(
        "Converting of {},{},{} took {:.8f}s".format(
            cube_xyz[0], cube_xyz[1], cube_xyz[2], time.time() - ref_time
        )
    )


def convert_knossos(
    source_path: str,
    target_path: str,
    layer_name: str,
    dtype: str,
    mag: int = 1,
    args: Namespace = None,
) -> None:
    source_knossos_info = KnossosDatasetInfo(source_path, dtype)
    target_wkw_info = WkwDatasetInfo(
        target_path, layer_name, mag, wkw.Header(convert_element_class_to_dtype(dtype))
    )

    ensure_wkw(target_wkw_info)

    with open_knossos(source_knossos_info) as source_knossos:
        with get_executor_for_args(args) as executor:
            knossos_cubes = list(source_knossos.list_cubes())
            if len(knossos_cubes) == 0:
                logging.error(
                    "No input KNOSSOS cubes found. Make sure to pass the path which points to a KNOSSOS magnification (e.g., testdata/knossos/color/1)."
                )
                exit(1)

            knossos_cubes.sort()
            job_args = []
            for cube_xyz in knossos_cubes:
                job_args.append((cube_xyz, source_knossos_info, target_wkw_info))

            wait_and_ensure_success(executor.map_to_futures(convert_cube_job, job_args))


if __name__ == "__main__":
    args = create_parser().parse_args()
    setup_logging(args)

    convert_knossos(
        args.source_path, args.target_path, args.layer_name, args.dtype, args.mag, args
    )
