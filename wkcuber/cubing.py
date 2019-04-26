import time
import logging
import numpy as np
from argparse import ArgumentParser
from os import path
import cluster_tools

from .utils import (
    get_chunks,
    find_files,
    add_verbose_flag,
    open_wkw,
    ensure_wkw,
    WkwDatasetInfo,
    add_distribution_flags,
    get_executor_for_args,
    wait_and_ensure_success,
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
        type=int,
        default=BLOCK_LEN,
    )

    add_verbose_flag(parser)
    add_distribution_flags(parser)

    return parser


def find_source_filenames(source_path):
    # Find all files in a folder that have a matching file extension
    source_files = list(
        find_files(path.join(source_path, "*"), image_reader.readers.keys())
    )
    assert len(source_files) > 0, (
        "No image files found in path "
        + source_path
        + ". Supported suffixes are "
        + str(image_reader.readers.keys())
        + "."
    )
    source_files.sort()
    return source_files


def read_image_file(file_name, dtype):
    try:
        return image_reader.read_array(file_name, dtype)
    except Exception as exc:
        logging.error("Reading of file={} failed with {}".format(file_name, exc))
        raise exc


def cubing_job(
    target_wkw_info,
    z_batches,
    source_file_batches,
    batch_size,
    image_size,
    num_channels,
):
    if len(z_batches) == 0:
        return

    with open_wkw(target_wkw_info, num_channels=num_channels) as target_wkw:
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
                    # Image shape will be (x, y, channel_count, z=1) or (x, y, z=1)
                    image = read_image_file(file_name, target_wkw_info.dtype)
                    assert (
                        image.shape[0:2] == image_size
                    ), "Section z={} has the wrong dimensions: {} (expected {}).".format(
                        z, image.shape, image_size
                    )
                    buffer.append(image)

                # Write batch buffer which will have shape (x, y, channel_count, z)
                # since we concat along the last axis (z)
                buffer = np.concatenate(buffer, axis=-1)

                if buffer.ndim == 4:
                    # In case of multi-channel data, we transpose the data
                    # so that the first dimension is the channel, since the wkw
                    # lib expects this.
                    # New shape will be (channel_count, x, y, z)
                    buffer = np.transpose(buffer, (2, 0, 1, 3))
                    assert buffer.shape[0] == num_channels

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


def cubing(source_path, target_path, layer_name, dtype, batch_size, args=None) -> dict:

    target_wkw_info = WkwDatasetInfo(target_path, layer_name, dtype, 1)
    source_files = find_source_filenames(source_path)

    # All images are assumed to have equal dimensions
    num_x, num_y = image_reader.read_dimensions(source_files[0])
    num_channels = image_reader.read_channel_count(source_files[0])
    num_z = len(source_files)

    logging.info("Found source files: count={} size={}x{}".format(num_z, num_x, num_y))

    ensure_wkw(target_wkw_info, num_channels=num_channels)

    with get_executor_for_args(args) as executor:
        futures = []
        # We iterate over all z sections
        for z in range(0, num_z, BLOCK_LEN):
            # Prepare z batches
            max_z = min(num_z, z + BLOCK_LEN)
            z_batch = list(range(z, max_z))
            # Execute
            futures.append(
                executor.submit(
                    cubing_job,
                    target_wkw_info,
                    z_batch,
                    source_files[z:max_z],
                    batch_size,
                    (num_x, num_y),
                    num_channels,
                )
            )

        wait_and_ensure_success(futures)

    # Return Bounding Box
    return {"topLeft": [0, 0, 0], "width": num_x, "height": num_y, "depth": num_z}


if __name__ == "__main__":
    args = create_parser().parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    cubing(
        args.source_path,
        args.target_path,
        args.layer_name,
        args.dtype,
        args.batch_size,
        args=args,
    )
