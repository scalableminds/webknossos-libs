import time
import logging
import numpy as np
import re
import wkw
from argparse import ArgumentParser
from os import path, listdir
from PIL import Image
from natsort import natsorted

from .utils import (
    get_chunks,
    get_regular_chunks,
    find_files,
    ensure_wkw,
    open_wkw,
    WkwDatasetInfo,
    add_distribution_flags,
    get_executor_for_args,
    wait_and_ensure_success,
    setup_logging,
)
from .cubing import create_parser, read_image_file, prepare_slices_for_wkw
from .image_readers import image_reader

BLOCK_LEN = 32
CUBE_REGEX = re.compile("(\d+)/(\d+)\.([a-zA-Z]{3,4})$")


def find_source_sections(source_path):
    section_folders = [
        f for f in listdir(source_path) if path.isdir(path.join(source_path, f))
    ]
    section_folders = [path.join(source_path, s) for s in section_folders]
    return natsorted(section_folders)


def parse_tile_file_name(filename):
    m = CUBE_REGEX.search(filename)
    if m is None:
        return None
    return (int(m.group(1)), int(m.group(2)), m.group(3))


def find_source_files(source_section_path):
    all_source_files = [
        path.join(source_section_path, s)
        for s in find_files(
            path.join(source_section_path, "**", "*"), image_reader.readers.keys()
        )
    ]
    return natsorted(all_source_files)


def tile_cubing_job(args):
    target_wkw_info, z_batches, source_path, batch_size, tile_size = args
    if len(z_batches) == 0:
        return

    with open_wkw(target_wkw_info) as target_wkw:
        # Iterate over the z batches
        # Batching is useful to utilize IO more efficiently
        for z_batch in get_chunks(z_batches, batch_size):
            try:
                ref_time = time.time()
                logging.info("Cubing z={}-{}".format(z_batch[0], z_batch[-1]))

                # Detect all available tiles in this z batch
                tile_coords = []
                for z in z_batch:
                    tile_coords += [
                        (z,) + parse_tile_file_name(f)
                        for f in find_source_files("{}/{}".format(source_path, z))
                    ]
                # Figure out what x-y combinations are available in this z batch
                xy = sorted(set([(x, y) for z, y, x, _ in tile_coords]))

                # Iterate over all x-y combinations from this z batch
                for x, y in xy:
                    ref_time2 = time.time()
                    slices = []
                    for z in z_batch:
                        # Find the file extension of the x-y tile in this z
                        ext = next(
                            (
                                ext
                                for _z, _y, _x, ext in tile_coords
                                if _z == z and _y == y and _x == x
                            ),
                            None,
                        )
                        # Read file if exists or zeros instead
                        if ext is not None:
                            image = read_image_file(
                                "{}/{}/{}/{}.{}".format(source_path, z, y, x, ext),
                                target_wkw_info.dtype,
                            )
                            slices.append(image)
                        else:
                            slices.append(
                                np.zeros(
                                    tile_size + (1, 1), dtype=target_wkw_info.dtype
                                )
                            )

                    # Write buffer to target
                    buffer = prepare_slices_for_wkw(slices)

                    if np.any(buffer != 0):
                        target_wkw.write(
                            [x * tile_size[0], y * tile_size[1], z_batch[0]], buffer
                        )
                    logging.debug(
                        "Cubing of z={}-{} x={} y={} took {:.8f}s".format(
                            z_batch[0], z_batch[-1], x, y, time.time() - ref_time2
                        )
                    )

                logging.debug(
                    "Cubing of z={}-{} took {:.8f}s".format(
                        z_batch[0], z_batch[-1], time.time() - ref_time
                    )
                )

            except Exception as exc:
                logging.error(
                    "Cubing of z={}-{} failed with {}".format(
                        z_batch[0], z_batch[-1], exc
                    )
                )
                raise exc


def tile_cubing(
    source_path, target_path, layer_name, dtype, batch_size, jobs, args=None
):
    # Detect available z sections
    sections = find_source_sections(source_path)
    if len(sections) == 0:
        logging.error("No source files found")
        return
    min_z = min([int(path.basename(f)) for f in sections])
    max_z = max([int(path.basename(f)) for f in sections])

    # Determine tile size from first matching file
    tile_size = image_reader.read_dimensions(
        next(find_files(path.join(source_path, "**", "*"), image_reader.readers.keys()))
    )
    logging.info(
        "Found source files: count={} tile_size={}x{}".format(
            len(sections), tile_size[0], tile_size[1]
        )
    )

    target_wkw_info = WkwDatasetInfo(target_path, layer_name, dtype, 1)
    ensure_wkw(target_wkw_info)
    with get_executor_for_args(args) as executor:
        job_args = []
        # Iterate over all z batches
        for z_batch in get_regular_chunks(min_z, max_z, BLOCK_LEN):
            job_args.append(
                (
                    target_wkw_info,
                    list(z_batch),
                    source_path,
                    int(batch_size),
                    tile_size,
                )
            )
        wait_and_ensure_success(executor.map_to_futures(tile_cubing_job, job_args))


if __name__ == "__main__":
    args = create_parser().parse_args()
    setup_logging(args)

    tile_cubing(
        args.source_path,
        args.target_path,
        args.layer_name,
        args.dtype,
        int(args.batch_size),
        int(args.jobs),
        args,
    )
