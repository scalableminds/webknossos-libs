import time
import logging
import numpy as np
import wkw
from argparse import ArgumentParser
from os import path, listdir
from PIL import Image

SOURCE_FORMAT_FILES = ('.tif', '.tiff', '.jpg', '.jpeg', '.png')


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


def open_wkw(_path, layer_name, dtype, mag):
    return wkw.Dataset.open(path.join(_path, layer_name, str(mag)), wkw.Header(np.dtype(dtype)))


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

    with open_wkw(args.target_path, args.layer_name, args.dtype, 1) as cube_io:

        source_files = find_source_filenames(args.source_path)
        num_x, num_y, num_z = determine_source_dims_from_images(
            source_files)

        # we iterate over the z layers
        for z in range(0, num_z):
            logging.info("Cubing z={0}".format(z))
            ref_time = time.time()

            this_layer = np.array(Image.open(source_files[z]))
            this_layer = this_layer.swapaxes(0, 1)
            this_layer = this_layer.reshape(this_layer.shape + (1,))

            cube_io.write([0, 0, z], this_layer)

            logging.debug("Cubing took {:.8f}s".format(time.time() - ref_time))
