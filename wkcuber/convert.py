import time
import logging
import wkw
import numpy as np
from argparse import ArgumentParser
from os import path

from .utils import \
    add_jobs_flag, add_verbose_flag, \
    open_wkw, open_knossos, \
    WkwDatasetInfo, KnossosDatasetInfo, ParallelExecutor
from .knossos import KnossosDataset, CUBE_EDGE_LEN


def create_parser():
    parser = ArgumentParser()

    parser.add_argument(
        'source_path',
        help="Directory containing the source KNOSSOS dataset.")

    parser.add_argument(
        'target_path',
        help="Output directory for the generated WKW dataset.")

    parser.add_argument(
        '--layer_name', '-l',
        help="Name of the cubed layer (color or segmentation)",
        default="color")

    parser.add_argument(
        '--dtype', '-d',
        help="Target datatype (e.g. uint8, uint16, uint32)",
        default="uint8")

    parser.add_argument(
        '--mag', '-m',
        help="Magnification level",
        default=1)

    add_jobs_flag(parser)
    add_verbose_flag(parser)

    return parser


def convert_cube_job(cube_xyz, source_knossos_info, target_wkw_info):
    logging.info("Converting {},{},{}".format(
        cube_xyz[0], cube_xyz[1], cube_xyz[2]))
    ref_time = time.time()
    offset = tuple(x * CUBE_EDGE_LEN for x in cube_xyz)
    size = (CUBE_EDGE_LEN, ) * 3

    with open_knossos(source_knossos_info) as source_knossos, \
            open_wkw(target_wkw_info) as target_wkw:
        cube_data = source_knossos.read(offset, size)
        target_wkw.write(offset, cube_data)
    logging.debug("Converting of {},{},{} took {:.8f}s".format(
        cube_xyz[0], cube_xyz[1], cube_xyz[2], time.time() - ref_time))


if __name__ == '__main__':
    args = create_parser().parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    source_knossos_info = KnossosDatasetInfo(
        args.source_path, args.layer_name, args.dtype, args.mag)
    target_wkw_info = WkwDatasetInfo(
        args.target_path, args.layer_name, args.dtype, args.mag)

    with open_knossos(source_knossos_info) as source_knossos, \
            ParallelExecutor(args.jobs) as pool:
        for cube_xyz in source_knossos.list_cubes():
            pool.submit(convert_cube_job, cube_xyz,
                        source_knossos_info, target_wkw_info)
