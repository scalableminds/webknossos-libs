import time
import logging
import numpy as np
import wkw
from argparse import ArgumentParser
from os import path, listdir
from PIL import Image

from .utils import (
    get_chunks,
    find_files,
    add_verbose_flag,
    add_jobs_flag,
    open_wkw,
    WkwDatasetInfo,
    ParallelExecutor,
    pool_get_lock,
)
from .image_readers import image_reader

BLOCK_LEN = 32


def create_parser():
    parser = ArgumentParser()

    parser.add_argument("source_path", help="Directory containing the input images.")

    parser.add_argument(
        "target_path", help="Output directory for the generated dataset."
    )

    parser.add_argument(
        "--layer_name",
        "-l",
        help="Name of the cubed layer (color or segmentation)",
        default="color",
    )

    parser.add_argument(
        "--dtype",
        "-d",
        help="Target datatype (e.g. uint8, uint16, uint32)",
        default="uint8",
    )

    parser.add_argument(
        "--batch_size",
        "-b",
        help="Number of sections to buffer per job",
        default=BLOCK_LEN,
    )

    add_verbose_flag(parser)
    add_jobs_flag(parser)

    return parser


def find_source_filenames(source_path):
    # Find all files in a folder that have a matching file extension
    source_files = list(
        find_files(path.join(source_path, "*"), image_reader.readers.keys())
    )
    source_files.sort()
    return source_files


def read_image_file(file_name, dtype):
    try:
        return image_reader.read_array(file_name, dtype)
    except Exception as exc:
        logging.error("Reading of file={} failed with {}".format(file_name, exc))
        raise exc


def cubing_job(target_wkw_info, z_batches, source_file_batches, batch_size, image_size):
    if len(z_batches) == 0:
        return

    with open_wkw(target_wkw_info, pool_get_lock()) as target_wkw:
        # Iterate over batches of continuous z sections
        # The batches have a maximum size of `batch_size`
        # Batched iterations allows to utilize IO more efficiently
        for z_batch, source_file_batch in zip(
            get_chunks(z_batches, batch_size),
            get_chunks(source_file_batches, batch_size),
        ):
            try:
                ref_time = time.time()
                logging.info("Cubing z={}-{}".format(z_batch[0], z_batch[-1]))
                buffer = []
                # Iterate over each z section in the batch
                for z, file_name in zip(z_batch, source_file_batch):
                    image = read_image_file(file_name, target_wkw_info.dtype)
                    assert (
                        image.shape[0:2] == image_size
                    ), "Section z={} has the wrong dimensions: {} (expected {}).".format(
                        z, image.shape, image_size
                    )
                    buffer.append(image)

                # Write batch buffer
                buffer = np.dstack(buffer)
                target_wkw.write([0, 0, z_batch[0]], buffer)
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


def cubing(source_path, target_path, layer_name, dtype, batch_size, jobs):
    target_wkw_info = WkwDatasetInfo(target_path, layer_name, dtype, 1)
    source_files = find_source_filenames(source_path)

    # All images are assumed to have equal dimensions
    num_x, num_y = image_reader.read_dimensions(source_files[0])
    num_z = len(source_files)

    logging.info("Found source files: count={} size={}x{}".format(num_z, num_x, num_y))
    with ParallelExecutor(jobs) as pool:
        # We iterate over all z sections
        for z in range(0, num_z, BLOCK_LEN):
            # Prepare z batches
            max_z = min(num_z, z + BLOCK_LEN)
            z_batch = list(range(z, max_z))
            # Execute
            pool.submit(
                cubing_job,
                target_wkw_info,
                z_batch,
                source_files[z:max_z],
                int(batch_size),
                (num_x, num_y),
            )


if __name__ == "__main__":
    args = create_parser().parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    cubing(
        args.source_path,
        args.target_path,
        args.layer_name,
        args.dtype,
        int(args.batch_size),
        int(args.jobs),
    )
