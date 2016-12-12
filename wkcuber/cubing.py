import time
import logging
import sys
import numpy as np
from math import log2, ceil
from os import path, listdir
from itertools import product
from PIL import Image
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from .utils import chunks
from .cube_io import write_cube, get_cube_folder

SOURCE_FORMAT_FILES = ('.tif', '.tiff', '.jpg', '.jpeg', '.png')

CubingInfo = namedtuple('CubingInfo',
                        'source_files source_dims cube_dims bbox resolutions')


def find_source_filenames(source_path):
    source_files = [
        f for f in listdir(source_path)
        if any([f.endswith(suffix) for suffix in SOURCE_FORMAT_FILES])]

    all_source_files = [path.join(source_path, s) for s in source_files]

    if len(all_source_files) == 0:
        logging.critical("No image files of format " +
                         source_format + " was found.")
        sys.exit()

    all_source_files.sort()
    return all_source_files


def determine_bbox(cube_dims, cube_edge_len):
    return tuple(map(lambda x: (x + 1) * cube_edge_len, cube_dims))


def determine_source_dims(source_files):
    # open the first image and extract the relevant information
    # all images are assumed to have equal dimensions!
    with Image.open(source_files[0]) as test_img:
        return (test_img.width, test_img.height, len(source_files))


def determine_cube_dims(source_dims, cube_edge_len):
    def cube_dim_mapper(dim):
        return int(ceil(dim / float(cube_edge_len)))
    return tuple(map(cube_dim_mapper, source_dims))


def determine_resolutions(cube_dims):
    max_mag = ceil(log2(max(cube_dims))) + 1
    return tuple(map(lambda x: 2 ** x, range(0, max_mag)))


def get_cubing_info(config):
    """Compute certain cubing parameters from the set of parameters
    specified by the user.
    Args:
        config (ConfigParser):
    Returns:
        source_files ([str*])
        source_dims (int, int, int)
        cube_dims (int, int, int)
        num_x_cubes_per_pass (int)
        num_passes_per_cube_layer (int)
    """
    source_path = config['dataset']['source_path']
    cube_edge_len = config['processing']['cube_edge_len']
    buffer_size_in_cubes = config['processing']['buffer_size_in_cubes']

    source_files = find_source_filenames(source_path)
    source_dims = determine_source_dims(source_files)
    cube_dims = determine_cube_dims(source_dims, cube_edge_len)
    bbox = determine_bbox(cube_dims, cube_edge_len)
    resolutions = determine_resolutions(cube_dims)

    return CubingInfo(source_files, source_dims, cube_dims, bbox, resolutions)


def check_layer_already_cubed(target_path, cur_z):
    folder = get_cube_folder(target_path, 1, 1, 1, cur_z)
    try:
        return any([file for file in listdir(folder) if file.endswith(".raw")])
    except FileNotFoundError:
        return False


def make_mag1_cubes_from_z_stack(config, cubing_info):

    source_files = cubing_info.source_files
    source_dims = cubing_info.source_dims
    cube_dims = cubing_info.cube_dims
    dtype = config['dataset']['dtype']
    target_path = config['dataset']['target_path']
    num_io_threads = config['processing']['num_io_threads']
    skip_already_cubed_layers = config[
        'processing']['skip_already_cubed_layers']
    cube_edge_len = config['processing']['cube_edge_len']
    buffer_size_in_cubes = config['processing']['buffer_size_in_cubes']
    num_x_cubes, num_y_cubes, num_z_cubes = cube_dims

    # we iterate over the z cubes and handle cube layer after cube layer
    for cube_z in range(0, num_z_cubes + 1):
        logging.info("Cubing layer: {0}".format(cube_z))

        if skip_already_cubed_layers and \
                check_layer_already_cubed(target_path, cube_z):
            logging.info("Skipping cube layer: {0}".format(cube_z))
            continue

        cube_coordinates = list(
            product(range(num_x_cubes), range(num_y_cubes)))

        for chunk_cube_coordinates in \
                chunks(cube_coordinates, buffer_size_in_cubes):
            cube_buffer = np.zeros((
                len(chunk_cube_coordinates),
                cube_edge_len,
                cube_edge_len,
                cube_edge_len),
                dtype=dtype)

            for local_z in range(0, cube_edge_len):
                z = cube_z * cube_edge_len + local_z
                try:
                    logging.debug("Loading {0}".format(source_files[z]))
                except IndexError:
                    logging.info("No more image files available.")
                    break

                ref_time = time.time()

                this_layer = np.array(Image.open(source_files[z]))
                this_layer = this_layer.swapaxes(0, 1)

                for i, (cube_x, cube_y) in enumerate(chunk_cube_coordinates):
                    cube_slice = this_layer[cube_x * cube_edge_len:
                                            (cube_x + 1) * cube_edge_len,
                                            cube_y * cube_edge_len:
                                            (cube_y + 1) * cube_edge_len]
                    cube_buffer[i, local_z] = cube_slice

                logging.debug("Reading took {:.8f}s".format(
                    time.time() - ref_time))

            # with ThreadPoolExecutor(max_workers=num_io_threads) as pool:
            # write out the cubes for this z-cube layer and buffer
            for i, (cube_x, cube_y) in enumerate(chunk_cube_coordinates):
                cube_data = cube_buffer[i].swapaxes(0, 1).swapaxes(1, 2)
                # pool.submit(write_cube, cube_data,
                #             target_path, 1, cube_x, cube_y, cube_z)
                write_cube(target_path, cube_data, 1, cube_x, cube_y, cube_z)
                logging.info("Cube written: {},{},{} mag {}".format(
                    cube_x, cube_y, cube_z, 1))
