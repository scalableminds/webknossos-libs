import time
import logging
import wkw
import re
import numpy as np
from argparse import ArgumentParser
from math import floor
from os import path, listdir
from scipy.ndimage.interpolation import zoom
from itertools import product

CUBE_EDGE_LEN = 128


def create_parser():
    parser = ArgumentParser()

    parser.add_argument(
        'path',
        help="Directory containing the dataset.")

    parser.add_argument(
        '--layer_name', '-l',
        help="Name of the cubed layer (color or segmentation)",
        default="color")

    parser.add_argument(
        '--segmentation', '-s',
        help="This layer is a segmentation layer",
        dest="segmentation",
        action='store_true')

    parser.add_argument(
        '--dtype', '-d',
        help="Target datatype (e.g. uint8, uint16, uint32)",
        default="uint8")

    parser.add_argument(
        '--max', '-m',
        help="Max resolution to be downsampled",
        default=512)

    parser.add_argument(
        '--verbose', '-v',
        help="Verbose output",
        dest="verbose",
        action='store_true')

    parser.set_defaults(verbose=False)
    parser.set_defaults(segmentation=False)

    return parser


def cube_addresses(source_wkw):

    wkw_cubelength = source_wkw.header.file_len * source_wkw.header.block_len
    factor = wkw_cubelength // CUBE_EDGE_LEN

    def parse_cube_file_name(filename):
        CUBE_REGEX = re.compile("z(\d+)/y(\d+)/x(\d+)(\.wkw)$")
        m = CUBE_REGEX.search(filename)
        return (int(m.group(3)), int(m.group(2)), int(m.group(1)))

    wkw_addresses = list(parse_cube_file_name(f)
                         for f in source_wkw.list_files())

    cube_addresses = []
    for wkw_x, wkw_y, wkw_z in wkw_addresses:
        x_dims = list(range(wkw_x * factor, (wkw_x + 1) * factor))
        y_dims = list(range(wkw_y * factor, (wkw_y + 1) * factor))
        z_dims = list(range(wkw_z * factor, (wkw_z + 1) * factor))
        cube_addresses += product(x_dims, y_dims, z_dims)

    return cube_addresses


def downsample(source_wkw, target_wkw, source_mag, target_mag, is_segmentation):
    assert source_mag < target_mag
    logging.info("Downsampling mag {} from mag {}".format(
        target_mag, source_mag))

    factor = int(target_mag / source_mag)
    cube_coordinates = list(set(tuple(x // factor for x in xyz)
                                for xyz in cube_addresses(source_wkw)))
    cube_coordinates.sort()

    for cube_x, cube_y, cube_z in cube_coordinates:
        downsample_cube_job(source_wkw, target_wkw,
                            factor, is_segmentation,
                            cube_x, cube_y, cube_z)


def downsample_cube_job(source_wkw, target_wkw, factor, is_segmentation,
                        cube_x, cube_y, cube_z):

    interpolation_order = 0 if is_segmentation else 1

    logging.debug("Downsampling {},{},{}".format(
        cube_x, cube_y, cube_z))

    source_offset = tuple(a * CUBE_EDGE_LEN for a in (cube_x, cube_y, cube_z))
    target_offset = tuple(a // factor for a in source_offset)

    ref_time = time.time()
    cube_buffer = source_wkw.read(source_offset, (CUBE_EDGE_LEN,) * 3)[0]
    if np.all(cube_buffer == 0):
        logging.debug("Skipping empty cube {},{},{}".format(
            cube_x, cube_y, cube_z))
    cube_data = downsample_cube(cube_buffer, factor, cube_buffer.dtype,
                                interpolation_order)
    target_wkw.write(target_offset, cube_data)

    logging.debug("Downsampling took {:.8f}s".format(
        time.time() - ref_time))


def downsample_cube(cube_buffer, factor, dtype, order):
    BILINEAR = 1
    BICUBIC = 2
    return zoom(
        cube_buffer, 1 / factor, output=dtype,
        # 0: nearest
        # 1: bilinear
        # 2: bicubic
        order=order,
        # this does not mean nearest interpolation, it corresponds to how the
        # borders are treated.
        mode='nearest',
        prefilter=True)


def open_wkw(dataset_path, layer_name, dtype, mag):
    return wkw.Dataset.open(path.join(dataset_path, layer_name, str(mag)), wkw.Header(np.dtype(dtype)))


if __name__ == '__main__':
    args = create_parser().parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    target_mag = 2
    while target_mag <= int(args.max):
        source_mag = target_mag // 2
        with open_wkw(args.path, args.layer_name, args.dtype, source_mag) as source_wkw, \
                open_wkw(args.path, args.layer_name, args.dtype, target_mag) as target_wkw:
            downsample(source_wkw, target_wkw, source_mag,
                       target_mag, args.segmentation)
        logging.info("Mag {0} succesfully cubed".format(target_mag))
        target_mag = target_mag * 2
