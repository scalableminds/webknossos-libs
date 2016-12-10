import threading
import io
import math
import scipy.ndimage
import numpy as np
import os
import itertools
import time
import sys
import re
import argparse
import yaml
import json
from os import path
from PIL import Image
from collections import OrderedDict, namedtuple
from concurrent.futures import ThreadPoolExecutor
from utils import chunks
import logging

SOURCE_FORMAT_FILES = ('.tif', '.tiff', '.jpg', '.jpeg', '.png')

class InvalidCubingConfigError(Exception):
    """An Exception object that is thrown whenever something goes wrong during
    cubing.
    """
    pass


CubingInfo = namedtuple('CubingInfo',
                        'source_files source_dims cube_dims bbox resolutions')

def determine_bbox(cube_dims, cube_edge_len):
    return tuple(map(lambda x: (x + 1) * cube_edge_len, cube_dims))


def write_webknossos_metadata(dataset_base_path,
                              name, 
                              scale, 
                              bbox,
                              resolutions):
    
    if not path.exists(dataset_base_path):
        os.makedirs(dataset_base_path)
    with open(path.join(dataset_base_path, 'settings.json'), 'wt') as settings_json:
        json.dump({
            'name': name,
            'priority': 0,
            'scale' : scale
        }, settings_json)

    if not path.exists(path.join(dataset_base_path, 'color')):
        os.makedirs(path.join(dataset_base_path, 'color'))
    with open(path.join(dataset_base_path, 'color', 'layer.json'), 'wt') as layer_json:
        json.dump({
            'typ' : 'color',
            'class': 'uint8'
        }, layer_json)

    with open(path.join(dataset_base_path, 'color', 'section.json'), 'wt') as section_json:
        json.dump({
            'bbox': (
                (0, bbox[0]),
                (0, bbox[1]),
                (0, bbox[2])
            ),
            'resolutions': resolutions
        }, section_json)


def find_source_filenames(source_path):
    source_files = [
        f for f in os.listdir(source_path)
        if any([f.endswith(suffix) for suffix in SOURCE_FORMAT_FILES])]

    all_source_files = [path.join(source_path, s) for s in source_files]

    if len(all_source_files) == 0:
        logging.critical("No image files of format " + source_format + " was found.")
        sys.exit()

    all_source_files.sort()
    return all_source_files


def determine_source_dims(source_files):
    # open the first image and extract the relevant information - all images are
    # assumed to have equal dimensions!
    with Image.open(source_files[0]) as test_img:
        return (test_img.width, test_img.height, len(source_files))


def determine_cube_dims(source_dims, cube_edge_len):
    # determine the number of passes required for each cube layer - if several
    # passes are required, we split the xy plane up in X cube_edge_len chunks,
    # always with full y height
    return tuple(map(lambda dim: int(math.ceil(dim / float(cube_edge_len))), source_dims))


def determine_resolutions(cube_dims):
    max_mag = math.ceil(math.log2(max(cube_dims))) + 1
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
        return any([file for file in os.listdir(folder) if file.endswith(".raw")])
    except FileNotFoundError:
        return False


def get_cube_folder(target_path, mag, x, y, z):
    return path.join(target_path, 'color', str(mag), 
        'x{:04d}'.format(x), 
        'y{:04d}'.format(y), 
        'z{:04d}'.format(z))


def get_cube_file_name(mag, x, y, z):
    return '{:s}_mag{:d}_x{:04d}_y{:04d}_z{:04d}.raw'.format('', mag, x, y, z)


def write_cube(target_path, cube_data, mag, x, y, z):
    ref_time = time.time()

    prefix = get_cube_folder(target_path, mag, x, y, z)
    file_name = get_cube_file_name(mag, x, y, z)
    cube_full_path = path.join(prefix, file_name)

    if not path.exists(prefix):
        os.makedirs(prefix)

    logging.debug("Writing cube {0}".format(cube_full_path))

    try:
        with open(cube_full_path, "wb") as cube_file:
          cube_data.ravel(order="F").tofile(cube_file)
        logging.debug("writing took: {:.8f}s".format(time.time() - ref_time))
    except IOError:
        logging.error("Could not write cube: {0}".format(cube_full_path))


def read_cube(target_path, mag, cube_edge_len, x, y, z):
    ref_time = time.time()

    prefix = get_cube_folder(target_path, mag, x, y, z)
    file_name = get_cube_file_name(mag, x, y, z)
    cube_full_path = path.join(prefix, file_name)

    if not path.exists(prefix):
        logging.debug("Missed cube {0}".format(cube_full_path))
        return np.zeros((cube_edge_len,) * 3, np.uint8)

    logging.debug("Reading cube {0}".format(cube_full_path))

    try:
        with open(cube_full_path, "rb") as cube_file:
          cube_data = np.fromfile(cube_file, dtype=np.uint8).reshape((cube_edge_len,) * 3, order="F")
        logging.debug("Reading took: {:.8f}s".format(time.time() - ref_time))
        return cube_data
    except IOError:
        logging.error("Could not read cube: {0}".format(cube_full_path))


def make_mag1_cubes_from_z_stack(config, source_files, source_dims, cube_dims):

    target_path = config['dataset']['target_path']
    num_io_threads = config['processing']['num_io_threads']
    skip_already_cubed_layers = config['processing']['skip_already_cubed_layers']
    cube_edge_len = config['processing']['cube_edge_len']
    buffer_size_in_cubes = config['processing']['buffer_size_in_cubes']
    num_x_cubes, num_y_cubes, num_z_cubes = cube_dims

    # we iterate over the z cubes and handle cube layer after cube layer
    for cube_z in range(0, num_z_cubes + 1):
        logging.info("Cubing layer: {0}".format(cube_z))

        if skip_already_cubed_layers and check_layer_already_cubed(target_path, cube_z):
            logging.debug("Skipping cube layer: {0}".format(cube_z))
            continue

        cube_coordinates = list(itertools.product(range(num_x_cubes), range(num_y_cubes)))
        
        for chunk_cube_coordinates in chunks(cube_coordinates, buffer_size_in_cubes):
            cube_buffer = np.zeros((
                len(chunk_cube_coordinates),
                cube_edge_len,
                cube_edge_len,
                cube_edge_len), 
            np.uint8)

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

                logging.debug("Reading took {:.8f}s".format(time.time() - ref_time))

            #with ThreadPoolExecutor(max_workers=num_io_threads) as pool:
            # write out the cubes for this z-cube layer and buffer
            for i, (cube_x, cube_y) in enumerate(chunk_cube_coordinates):
                # pool.submit(write_cube, cube_buffer[i], target_path, 1, cube_x, cube_y, cube_z)
                cube_data = cube_buffer[i].swapaxes(0, 1).swapaxes(1, 2)
                write_cube(target_path, cube_data, 1, cube_x, cube_y, cube_z)
                logging.info("Cube written: {},{},{} mag {}".format(cube_x, cube_y, cube_z, 1))


def knossos_cuber(config):
    cubing_info = get_cubing_info(config)

    mag1_ref_time = time.time()

    logging.info("Creating mag1 cubes.")

    make_mag1_cubes_from_z_stack(
        config,
        cubing_info.source_files,
        cubing_info.source_dims,
        cubing_info.cube_dims)

    write_webknossos_metadata(config['dataset']['target_path'], 
                              config['dataset']['name'], 
                              config['dataset']['scale'], 
                              cubing_info.bbox,
                              cubing_info.resolutions)

    logging.info("Mag 1 succesfully cubed. Took {:.3f}h".format((time.time() - mag1_ref_time) / 3600))

    total_down_ref_time = time.time()
    curr_mag = 2

    while curr_mag <= max(cubing_info.resolutions):
        downsample(config, curr_mag // 2, curr_mag)
        logging.info("Mag {0} succesfully cubed.".format(curr_mag))
        curr_mag = curr_mag * 2

    logging.info("All mags generated. Took {:.3f}h."
           .format((time.time() - total_down_ref_time)/3600))

    logging.info('All done.')


def create_parser():
    """Creates a parser for command-line arguments.
    The parser can read 4 options:
        Optional arguments:
            --config, -c : path to a configuration file
            --name, -n : dataset name
        Positional arguments:
            source_dir : path to input files
            target_dir : output path
    Args:
    Returns:
        An ArgumentParser object.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'source_dir',
        help="Directory containing the input images.")

    parser.add_argument(
        'target_dir',
        help="Output directory for the generated dataset.")

    parser.add_argument(
        '--name', '-n',
        help="Name of the dataset. If no name is specified, the source directory name will be used.")

    parser.add_argument(
        '--loglevel', '-ll',
        help="Loglevel. DEBUG, INFO, WARNING, ERROR or CRITICAL.",
        default='INFO')

    parser.add_argument(
        '--config', '-c',
        help="A configuration file. If no file is specified, `config.yml' "
             "from knossos_cuber's installation directory is used. Note that "
             "you still have to specify the input/output directory and "
             "source format via the command line.",
        default='config.yml')

    return parser


cube_folder_regex = re.compile('^[xyz]\d{4}$')

def determine_cube_dims2(target_path, mag):

    prefix = path.join(target_path, 'color', str(mag))
    

    max_x = len(list(filter(cube_folder_regex.match, 
                            os.listdir(prefix))))
    max_y = len(list(filter(cube_folder_regex.match, 
                            os.listdir(path.join(prefix, 'x0000')))))
    max_z = len(list(filter(cube_folder_regex.match, 
                            os.listdir(path.join(prefix, 'x0000', 'y0000')))))

    return (max_x, max_y, max_z)




def downsample(config, source_mag, target_mag):

    assert source_mag < target_mag
    logging.info("Downsampling mag {} from mag {}.".format(target_mag, source_mag))

    factor = int(target_mag / source_mag)
    target_path = config['dataset']['target_path']
    cube_edge_len = config['processing']['cube_edge_len']
    skip_already_downsampled_cubes = config['processing']['skip_already_downsampled_cubes']

    source_cube_dims = determine_cube_dims2(target_path, source_mag)
    target_cube_dims = tuple(map(lambda x: math.ceil(x / factor), source_cube_dims))

    cube_coordinates = itertools.product(
        range(target_cube_dims[0]), 
        range(target_cube_dims[1]), 
        range(target_cube_dims[2]))

    for cube_x, cube_y, cube_z in cube_coordinates:
        cube_full_path = path.join(get_cube_folder(target_path, target_mag, cube_x, cube_y, cube_z), 
                                   get_cube_file_name(target_mag, cube_x, cube_y, cube_z))
        if skip_already_downsampled_cubes and path.exists(cube_full_path):
            continue

        logging.debug("Downsampling {},{},{}".format(cube_x, cube_y, cube_z))
        
        ref_time = time.time()
        cube_buffer = np.zeros((cube_edge_len * factor,) * 3, np.uint8)
        for local_x in range(factor):
            for local_y in range(factor):
                for local_z in range(factor):
                    cube_buffer[
                        local_x * cube_edge_len:(local_x + 1) * cube_edge_len,
                        local_y * cube_edge_len:(local_y + 1) * cube_edge_len,
                        local_z * cube_edge_len:(local_z + 1) * cube_edge_len] = \
                        read_cube(target_path, source_mag, cube_edge_len,
                                  cube_x * factor + local_x, 
                                  cube_y * factor + local_y, 
                                  cube_z * factor + local_z)

        cube_data = downsample_cube(cube_buffer, factor)
        write_cube(target_path, cube_data, target_mag, cube_x, cube_y, cube_z)

        logging.debug("Downsampling took {:.8f}s".format(time.time() - ref_time))
        logging.info("Downsampled cube: {},{},{} mag {}".format(cube_x, cube_y, cube_z, target_mag))



def downsample_cube(cube_buffer, factor):

    return scipy.ndimage.interpolation.zoom(
        cube_buffer, 1 / factor, output=np.uint8,
        # 1: bilinear
        # 2: bicubic
        order=1,
        # this does not mean nearest interpolation, it corresponds to how the
        # borders are treated.
        mode='nearest',
        prefilter=True)


def main():
    PARSER = create_parser()
    ARGS = PARSER.parse_args()

    logging.basicConfig(level=getattr(logging, ARGS.loglevel.upper(), None))

    with open(ARGS.config, 'r') as config_yaml:
        CONFIG = yaml.load(config_yaml)

    CONFIG['dataset'] = {
        'source_path': ARGS.source_dir,
        'target_path': ARGS.target_dir,
        'name': ARGS.name,
        'scale': (12, 12, 25)
    }

    knossos_cuber(CONFIG)

if __name__ == '__main__':
    main()