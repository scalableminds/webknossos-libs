import time
import logging
import numpy as np
from os import path, makedirs


def get_cube_folder(target_path, mag, x, y, z):
    return path.join(target_path, 'color', str(mag),
                     'x{:04d}'.format(x),
                     'y{:04d}'.format(y),
                     'z{:04d}'.format(z))


def get_cube_file_name(mag, x, y, z):
    return '{:s}_mag{:d}_x{:04d}_y{:04d}_z{:04d}.raw'.format('', mag, x, y, z)


def get_cube_full_path(target_path, mag, x, y, z):
    return path.join(get_cube_folder(target_path, mag, x, y, z),
                     get_cube_file_name(mag, x, y, z))


def write_cube(target_path, cube_data, mag, x, y, z):
    ref_time = time.time()

    prefix = get_cube_folder(target_path, mag, x, y, z)
    file_name = get_cube_file_name(mag, x, y, z)
    cube_full_path = path.join(prefix, file_name)

    if not path.exists(prefix):
        makedirs(prefix)

    logging.debug("Writing cube {0}".format(cube_full_path))

    try:
        with open(cube_full_path, "wb") as cube_file:
            cube_data.ravel(order="F").tofile(cube_file)
        logging.debug("writing took: {:.8f}s".format(time.time() - ref_time))
    except IOError:
        logging.error("Could not write cube: {0}".format(cube_full_path))


def read_cube(target_path, mag, cube_edge_len, x, y, z, dtype):
    ref_time = time.time()

    prefix = get_cube_folder(target_path, mag, x, y, z)
    file_name = get_cube_file_name(mag, x, y, z)
    cube_full_path = path.join(prefix, file_name)

    if not path.exists(prefix):
        logging.debug("Missed cube {0}".format(cube_full_path))
        return np.zeros((cube_edge_len,) * 3, dtype=dtype)

    logging.debug("Reading cube {0}".format(cube_full_path))

    try:
        with open(cube_full_path, "rb") as cube_file:
            cube_data = np.fromfile(cube_file, dtype=dtype).reshape(
                (cube_edge_len,) * 3, order="F")
        logging.debug("Reading took: {:.8f}s".format(time.time() - ref_time))
        return cube_data
    except IOError:
        logging.error("Could not read cube: {0}".format(cube_full_path))
