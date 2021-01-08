import time
import logging
from typing import List, Tuple

import numpy as np
import wkw
from argparse import ArgumentParser, Namespace
from os import path
from natsort import natsorted

from .mag import Mag
from .downsampling import (
    parse_interpolation_mode,
    downsample_unpadded_data,
    InterpolationModes,
)
from .utils import (
    get_chunks,
    find_files,
    add_verbose_flag,
    add_batch_size_flag,
    open_wkw,
    ensure_wkw,
    WkwDatasetInfo,
    add_distribution_flags,
    add_interpolation_flag,
    get_executor_for_args,
    wait_and_ensure_success,
    setup_logging,
)
from .image_readers import image_reader
from .metadata import convert_element_class_to_dtype

BLOCK_LEN = 32


def create_parser() -> ArgumentParser:
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
        "--start_z", help="The z coordinate of the first slice", default=0, type=int
    )

    parser.add_argument(
        "--dtype",
        "-d",
        help="Target datatype (e.g. uint8, uint16, uint32)",
        default="uint8",
    )

    parser.add_argument(
        "--wkw_file_len",
        default=32,
        type=int,
        help="Amount of blocks which are written per dimension to a wkw cube. The default value of 32 means that 1024 slices are written to one cube (since one block has 32**3 voxels by default). For single-channel uint8 data, this results in 1 GB per cube file. If file_len is set to 1, only 32 slices are written to one cube. Must be a power of two.",
    )

    add_batch_size_flag(parser)

    parser.add_argument(
        "--pad",
        help="Automatically pad image files at the bottom and right borders. "
        "Use this, when the input images don't have a common size, but have "
        "their origin at (0, 0).",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--target_mag",
        help="Automatically downsamples the cubed images to the provided "
        "magnification before writing to disk. The magnification can "
        "be provided like 2-2-1.",
        default="1",
    )

    add_interpolation_flag(parser)
    add_verbose_flag(parser)
    add_distribution_flags(parser)

    return parser


def find_source_filenames(source_path: str) -> List[str]:
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
    return natsorted(source_files)


def read_image_file(file_name: str, dtype: type, z_slice: int) -> np.ndarray:
    try:
        return image_reader.read_array(file_name, dtype, z_slice)
    except Exception as exc:
        logging.error("Reading of file={} failed with {}".format(file_name, exc))
        raise exc


def prepare_slices_for_wkw(
    slices: List[np.ndarray], num_channels: int = None
) -> np.ndarray:
    # Write batch buffer which will have shape (x, y, channel_count, z)
    # since we concat along the last axis (z)
    buffer = np.concatenate(slices, axis=-1)

    # We transpose the data so that the first dimension is the channel,
    # since the wkw library expects this.
    # New shape will be (channel_count, x, y, z)
    buffer = np.transpose(buffer, (2, 0, 1, 3))
    if num_channels is not None:
        assert buffer.shape[0] == num_channels
    return buffer


def cubing_job(
    args: Tuple[
        WkwDatasetInfo,
        List[int],
        Mag,
        InterpolationModes,
        List[str],
        int,
        Tuple[int, int],
        bool,
    ]
) -> None:
    (
        target_wkw_info,
        z_batches,
        target_mag,
        interpolation_mode,
        source_file_batches,
        batch_size,
        image_size,
        pad,
    ) = args
    if len(z_batches) == 0:
        return

    downsampling_needed = target_mag != Mag(1)

    with open_wkw(target_wkw_info) as target_wkw:
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
                slices = []
                # Iterate over each z section in the batch
                for z, file_name in zip(z_batch, source_file_batch):
                    # Image shape will be (x, y, channel_count, z=1)
                    image = read_image_file(
                        file_name, target_wkw_info.header.voxel_type, z
                    )

                    if not pad:
                        assert (
                            image.shape[0:2] == image_size
                        ), "Section z={} has the wrong dimensions: {} (expected {}). Consider using --pad.".format(
                            z, image.shape, image_size
                        )
                    slices.append(image)

                if pad:
                    x_max = max(_slice.shape[0] for _slice in slices)
                    y_max = max(_slice.shape[1] for _slice in slices)

                    slices = [
                        np.pad(
                            _slice,
                            mode="constant",
                            pad_width=[
                                (0, x_max - _slice.shape[0]),
                                (0, y_max - _slice.shape[1]),
                                (0, 0),
                                (0, 0),
                            ],
                        )
                        for _slice in slices
                    ]

                buffer = prepare_slices_for_wkw(
                    slices, target_wkw_info.header.num_channels
                )
                if downsampling_needed:
                    buffer = downsample_unpadded_data(
                        buffer, target_mag, interpolation_mode
                    )

                target_wkw.write([0, 0, z_batch[0] / target_mag.to_array()[2]], buffer)
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


def cubing(
    source_path: str,
    target_path: str,
    layer_name: str,
    dtype: str,
    batch_size: int,
    args: Namespace,
) -> dict:
    source_files = find_source_filenames(source_path)

    # All images are assumed to have equal dimensions
    num_x, num_y = image_reader.read_dimensions(source_files[0])
    num_channels = image_reader.read_channel_count(source_files[0])
    num_z_slices_per_file = image_reader.read_z_slices_per_file(source_files[0])
    assert (
        num_z_slices_per_file == 1 or len(source_files) == 1
    ), "Multi page TIFF support only for single files"
    if num_z_slices_per_file > 1:
        num_z = num_z_slices_per_file
    else:
        num_z = len(source_files)

    target_mag = Mag(args.target_mag)
    target_wkw_info = WkwDatasetInfo(
        target_path,
        layer_name,
        target_mag,
        wkw.Header(
            convert_element_class_to_dtype(dtype),
            num_channels,
            file_len=args.wkw_file_len,
        ),
    )
    interpolation_mode = parse_interpolation_mode(
        args.interpolation_mode, target_wkw_info.layer_name
    )
    if target_mag != Mag(1):
        logging.info(
            f"Downsampling the cubed image to {target_mag} in memory with interpolation mode {interpolation_mode}."
        )

    logging.info("Found source files: count={} size={}x{}".format(num_z, num_x, num_y))

    ensure_wkw(target_wkw_info)

    start_z = args.start_z

    with get_executor_for_args(args) as executor:
        job_args = []
        # We iterate over all z sections
        for z in range(start_z, num_z + start_z, BLOCK_LEN):
            # Prepare z batches
            max_z = min(num_z + start_z, z + BLOCK_LEN)
            z_batch = list(range(z, max_z))
            # Prepare source files array
            if len(source_files) > 1:
                source_files_array = source_files[z - start_z : max_z - start_z]
            else:
                source_files_array = source_files * (max_z - z)
            # Prepare job
            job_args.append(
                (
                    target_wkw_info,
                    z_batch,
                    target_mag,
                    interpolation_mode,
                    source_files_array,
                    batch_size,
                    (num_x, num_y),
                    args.pad,
                )
            )

        wait_and_ensure_success(executor.map_to_futures(cubing_job, job_args))

    # Return Bounding Box
    return {"topLeft": [0, 0, 0], "width": num_x, "height": num_y, "depth": num_z}


if __name__ == "__main__":
    args = create_parser().parse_args()
    setup_logging(args)

    cubing(
        args.source_path,
        args.target_path,
        args.layer_name,
        args.dtype,
        args.batch_size,
        args=args,
    )
