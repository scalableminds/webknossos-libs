import time
import logging
import wkw
import numpy as np
from argparse import ArgumentParser
from os import path
from .knossos import KnossosDataset, CUBE_EDGE_LEN

def open_wkw(_path, layer_name, dtype, mag):
    return wkw.Dataset.open(path.join(_path, layer_name, str(mag)), wkw.Header(np.dtype(dtype)))

def open_knossos(_path, layer_name, dtype, mag):
    return KnossosDataset.open(path.join(_path, layer_name, str(mag)), np.dtype(dtype))


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

    parser.add_argument(
        '--verbose', '-v',
        help="Verbose output",
        dest="verbose",
        action='store_true')

    parser.set_defaults(verbose=False)

    return parser


if __name__ == '__main__':
    args = create_parser().parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    with open_knossos(args.source_path, args.layer_name, args.dtype, args.mag) as source_knossos, \
         open_wkw(args.target_path, args.layer_name, args.dtype, args.mag) as target_wkw:

        for cube_xyz in source_knossos.list_cubes():
            logging.info("Converting {},{},{}".format(
                cube_xyz[0], cube_xyz[1], cube_xyz[2]))
            ref_time = time.time()
            offset = tuple(x * CUBE_EDGE_LEN for x in cube_xyz)
            size = (CUBE_EDGE_LEN, ) * 3

            cube_data = source_knossos.read(offset, size)
            target_wkw.write(offset, cube_data)
            logging.debug("Converting took {:.8f}s".format(time.time() - ref_time))

