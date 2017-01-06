import time
import logging
import numpy as np
import glob
from os import path, makedirs


def get_cube_folder(target_path, layer_name, mag, x, y, z):
    return path.join(target_path, layer_name, str(mag),
                     'x{:04d}'.format(x),
                     'y{:04d}'.format(y),
                     'z{:04d}'.format(z))


def get_cube_file_name(ds_name, mag, x, y, z):
    return '{:s}_mag{:d}_x{:04d}_y{:04d}_z{:04d}.raw'.format(
        str(ds_name), mag, x, y, z)


def get_only_raw_file_path(target_path, layer_name, mag, x, y, z):

    cube_folder = get_cube_folder(target_path, layer_name, mag, x, y, z)
    raw_files = glob.glob(path.join(cube_folder, "*.raw"))

    if len(raw_files) > 1:
        raise ValueError("Found %d .raw files in %s" %
                         (len(raw_files), cube_folder))

    return raw_files[0] if len(raw_files) > 0 else None


def get_cube_full_path(target_path, ds_name, layer_name, mag, x, y, z):
    return path.join(get_cube_folder(target_path, layer_name, mag, x, y, z),
                     get_cube_file_name(ds_name, mag, x, y, z))


def write_cube(target_path, cube_data, ds_name, layer_name, mag, x, y, z):
    ref_time = time.time()

    cube_full_path = get_cube_full_path(target_path, ds_name, layer_name,
                                        mag, x, y, z)

    makedirs(path.dirname(cube_full_path), exist_ok=True)

    logging.debug("Writing cube {0}".format(cube_full_path))

    try:
        with open(cube_full_path, "wb") as cube_file:
            cube_data.ravel(order="F").tofile(cube_file)
        logging.debug("writing took: {:.8f}s".format(time.time() - ref_time))
    except IOError:
        logging.error("Could not write cube: {0}".format(cube_full_path))


def read_cube(target_path, layer_name, mag, cube_edge_len, x, y, z,
              dtype):
    ref_time = time.time()

    cube_full_path = get_only_raw_file_path(target_path, layer_name,
                                            mag, x, y, z)

    if cube_full_path is None:
        logging.debug("Missed cube: {},{},{} mag {}".format(
                        x, y, z, mag))
        return None

    logging.debug("Reading cube {0}".format(cube_full_path))

    try:
        with open(cube_full_path, "rb") as cube_file:
            cube_data = np.fromfile(cube_file, dtype=dtype).reshape(
                (cube_edge_len,) * 3, order="F")
        logging.debug("Reading took: {:.8f}s".format(time.time() - ref_time))
        return cube_data
    except IOError:
        logging.error("Could not read cube: {0}".format(cube_full_path))
