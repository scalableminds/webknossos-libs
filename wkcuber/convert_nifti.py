import time
import logging
import wkw
import numpy as np
from argparse import ArgumentParser
from os import path
import nibabel as nib

from .utils import (
    add_verbose_flag,
    open_wkw,
    WkwDatasetInfo,
    ensure_wkw,
    add_distribution_flags,
    get_executor_for_args,
    wait_and_ensure_success,
    setup_logging,
)
from .image_readers import to_target_datatype
from .metadata import write_webknossos_metadata


def create_parser():
    parser = ArgumentParser()

    parser.add_argument(
        "source_path", help="Directory containing the source NIFTI dataset."
    )

    parser.add_argument(
        "target_path", help="Output directory for the generated WKW dataset."
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

    add_verbose_flag(parser)
    add_distribution_flags(parser)

    return parser


def to_target_datatype(data: np.ndarray, target_dtype) -> np.ndarray:

    if data.dtype == np.dtype("float32") or data.dtype == np.dtype("float64"):
        factor = data.max()

    else:
        factor = 1 + np.iinfo(data.dtype).max

    if target_dtype != np.dtype("float32"):
        factor = factor / (1 + np.iinfo(target_dtype).max)
    return (data / factor).astype(np.dtype(target_dtype))


def convert_nifti(
    source_nifti_path, target_path, layer_name, dtype, mag=1
):
    target_wkw_info = WkwDatasetInfo(target_path, layer_name, dtype, mag)
    ensure_wkw(target_wkw_info)

    ref_time = time.time()
    # ignoring affine transformation
    offset = (0, 0, 0)

    with open_wkw(target_wkw_info) as target_wkw:
        source_nifti = nib.load(source_nifti_path)
        cube_data = np.array(source_nifti.get_fdata())
        if len(source_nifti.shape) == 3:
            size = list(source_nifti.shape)
            cube_data = cube_data.reshape((1,) + source_nifti.shape)


        elif len(source_nifti.shape) == 4:
            size = list(source_nifti.shape[:-1])
            cube_data = np.transpose(cube_data, (3, 0, 1, 2))

        else:
            logging.warning(
                "Converting of {} failed! Too many or too less dimensions".format(
                    source_nifti_path
                )
            )

            return

        cube_data = to_target_datatype(cube_data, dtype)
        target_wkw.write(offset, cube_data)


    logging.debug(
        "Converting of {} took {:.8f}s".format(
            source_nifti_path, time.time() - ref_time
        )
    )

    write_webknossos_metadata(
        target_path,
        source_nifti_path.split("/")[-2],
        scale="10,10,10",
        exact_bounding_box={"topleft": offset, "size": size}
    )


if __name__ == "__main__":
    args = create_parser().parse_args()
    setup_logging(args)

    convert_nifti(
        args.source_path,
        args.target_path,
        args.layer_name,
        args.dtype
    )
