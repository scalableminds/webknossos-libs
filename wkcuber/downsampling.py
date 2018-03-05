import time
import logging
import numpy as np
from math import floor
from os import path, listdir
from scipy.ndimage.interpolation import zoom
from concurrent.futures import ProcessPoolExecutor

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
        '--jobs', '-j',
        help="Number of parallel jobs",
        default=4)

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

def get_dataset_size(source_wkw):
    file_size = source_wkw.header.block_len * source_wkw.header.file_len
    return (file_size,) * 3

def cube_addresses(size):
    x_dims = list(range(0, size)).sort()
    y_dims.sort()
    z_dims.sort()

def downsample(source_wkw, target_wkw, source_mag, target_mag, num_downsampling_cores, is_segmentation):
    assert source_mag < target_mag
    logging.info("Downsampling mag {} from mag {}".format(
        target_mag, source_mag))

    factor = int(target_mag / source_mag)
    source_size = get_dataset_size(source_wkw)
    cube_coordinates = set(tuple(floor(x / factor) for x in xyz) for xyz in source_cube_dims)

    with ProcessPoolExecutor(num_downsampling_cores) as pool:
        logging.debug("Using up to {} worker processes".format(
            num_downsampling_cores))
        for cube_x, cube_y, cube_z in cube_coordinates:
            pool.submit(downsample_cube_job, 
                        source_wkw, target_wkw, 
                        factor, is_segmentation, 
                        cube_x, cube_y, cube_z)


def downsample_cube_job(source_wkw, target_wkw, factor, is_segmentation,
                        cube_x, cube_y, cube_z):

    interpolation_order = 0 if is_segmentation else 1

    logging.debug("Downsampling {},{},{}".format(
        cube_x, cube_y, cube_z))

    source_offset = tuple(a * CUBE_EDGE_LEN for a in (cube_x, cube_y, cube_z))
    target_offset = tuple(a / factor for a in source_offset)

    ref_time = time.time()
    cube_buffer = source_wkw.read(source_offset, (CUBE_EDGE_LEN,)*3)
    cube_data = downsample_cube(cube_buffer, factor, cube_buffer.dtype,
                                interpolation_order)
    target.write(target_offset, cube_data)

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

def open_wkw(path, layer_name, dtype, mag):
    return wkw.Dataset.open(path.join(path, layer_name, str(mag)), wkw.Header(np.dtype(dtype)))

if __name__ == '__main__':
    args = create_parser().parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    target_mag = 2
    while target_mag <= args.max:
        source_mag = target_mag // 2
        with open_wkw(args.path, args.layer_name, args.dtype, source_mag) as source_wkw, open_wkw(args.path, args.layer_name, args.dtype, target_mag) as target_wkw:
            downsample(source_wkw, target_wkw, source_mag, target_mag, args.cores, args.segmentation)
        logging.info("Mag {0} succesfully cubed".format(target_mag))
        target_mag = target_mag * 2
