import time
import logging
import numpy as np
import wkw
from argparse import ArgumentParser
from os import path, listdir
from PIL import Image

from .utils import \
    add_verbose_flag, add_jobs_flag, \
    open_wkw, WkwDatasetInfo, ParallelExecutor

SOURCE_FORMAT_FILES = ('.tif', '.tiff', '.jpg', '.jpeg', '.png')
BLOCK_LEN = 32


def find_source_filenames(source_path):
    source_files = [
        f for f in listdir(source_path)
        if any([f.endswith(suffix) for suffix in SOURCE_FORMAT_FILES])]

    all_source_files = [path.join(source_path, s) for s in source_files]

    all_source_files.sort()
    return all_source_files


def determine_source_dims_from_images(source_files):
    # open the first image and extract the relevant information
    # all images are assumed to have equal dimensions!
    with Image.open(source_files[0]) as test_img:
        return (test_img.width, test_img.height, len(source_files))


def create_parser():
    parser = ArgumentParser()

    parser.add_argument(
        'source_path',
        help="Directory containing the input images.")

    parser.add_argument(
        'target_path',
        help="Output directory for the generated dataset.")

    parser.add_argument(
        '--layer_name', '-l',
        help="Name of the cubed layer (color or segmentation)",
        default="color")

    parser.add_argument(
        '--dtype', '-d',
        help="Target datatype (e.g. uint8, uint16, uint32)",
        default="uint8")

    add_verbose_flag(parser)
    add_jobs_flag(parser)

    return parser


def cubing_job(target_wkw_info, z_slice, source_file_slice):
    logging.info(z_slice)
    with open_wkw(target_wkw_info) as target_wkw:
        for z, file_name in zip(z_slice, source_file_slice):
            logging.info("Cubing z={}".format(z))
            ref_time = time.time()

            this_layer = np.array(Image.open(file_name))
            this_layer = this_layer.swapaxes(0, 1)
            this_layer = this_layer.reshape(this_layer.shape + (1,))

            target_wkw.write([0, 0, z], this_layer)

            logging.debug("Cubing of {} took {:.8f}s".format(
                z, time.time() - ref_time))


if __name__ == '__main__':
    args = create_parser().parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    target_wkw_info = WkwDatasetInfo(
        args.target_path, args.layer_name, args.dtype, 1)
    source_files = find_source_filenames(args.source_path)
    num_x, num_y, num_z = determine_source_dims_from_images(source_files)

    with ParallelExecutor(args.jobs) as pool:
        # we iterate over the z layers
        for z in range(0, num_z, BLOCK_LEN):
            max_z = min(num_z, z + BLOCK_LEN)
            pool.submit(cubing_job, target_wkw_info, list(
                range(z, max_z)), source_files[z:max_z])
