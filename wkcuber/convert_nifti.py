import time
import logging
import wkw
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
import nibabel as nib

from .utils import (
    add_verbose_flag,
    open_wkw,
    WkwDatasetInfo,
    ensure_wkw,
    setup_logging,
    add_scale_flag,
)

from .metadata import write_webknossos_metadata


def create_parser():
    parser = ArgumentParser()

    parser.add_argument(
        "source_path",
        help="Path to NIFTY file or to a directory if multiple NIFTI files should be converted. "
        "In the latter case, also see --color_file and --segmentation_file.",
    )

    parser.add_argument(
        "target_path", help="Output directory for the generated WKW dataset."
    )

    parser.add_argument(
        "--layer_name",
        "-l",
        help="Name of the cubed layer (color or segmentation).",
        default="color",
    )

    parser.add_argument(
        "--dtype", "-d", help="Target datatype (e.g. uint8, uint16).", default="uint8"
    )

    parser.add_argument(
        "--color_file",
        help="When converting folder, name of file to become color layer",
        default=None,
    )

    parser.add_argument(
        "--segmentation_file",
        help="When converting folder, name of file to become segmentation layer",
        default=None,
    )

    add_scale_flag(parser, required=False)
    add_verbose_flag(parser)

    return parser


def to_target_datatype(data: np.ndarray, target_dtype) -> np.ndarray:
    if data.dtype == np.dtype("float32"):
        factor = data.max()
    elif data.dtype == np.dtype("float64"):
        factor = data.max() / np.iinfo(target_dtype).max
    else:
        factor = np.iinfo(data.dtype).max / np.iinfo(target_dtype).max

    return (data / factor).astype(np.dtype(target_dtype))


def convert_nifti(
    source_nifti_path, target_path, layer_name, dtype, scale, mag=1, file_len=256
):
    target_wkw_info = WkwDatasetInfo(
        str(target_path.resolve()),
        layer_name,
        mag,
        wkw.Header(
            np.dtype(dtype),
            block_type=wkw.Header.BLOCK_TYPE_LZ4HC,
            file_len=file_len // 32,
        ),
    )
    ensure_wkw(target_wkw_info)

    ref_time = time.time()
    # Assume no translation
    offset = (0, 0, 0)

    with open_wkw(target_wkw_info) as target_wkw:
        source_nifti = nib.load(str(source_nifti_path.resolve()))
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

        if scale is None:
            scale = tuple(map(float, source_nifti.header["pixdim"][:3]))

        cube_data = to_target_datatype(cube_data, dtype)

        # Writing wkw compressed requires files of shape (file_len, file_len, file_len)
        # Pad data accordingly
        padding_offset = file_len - np.array(cube_data.shape[1:4]) % file_len
        cube_data = np.pad(
            cube_data,
            (
                (0, 0),
                (0, int(padding_offset[0])),
                (0, int(padding_offset[1])),
                (0, int(padding_offset[2])),
            ),
        )

        target_wkw.write(offset, cube_data)

    logging.debug(
        "Converting of {} took {:.8f}s".format(
            source_nifti_path, time.time() - ref_time
        )
    )

    write_webknossos_metadata(
        str(target_path),
        source_nifti_path.stem,
        scale=scale,
        exact_bounding_box={
            "topLeft": offset,
            "width": size[0],
            "height": size[1],
            "depth": size[2],
        },
    )


def convert_folder_nifti(
    source_folder_path, target_path, color_path, segmentation_path, scale
):
    paths = list(source_folder_path.rglob("**/*.nii"))

    if color_path not in paths and color_path is not None:
        logging.warning(
            "Specified color file {} not in source path {}!".format(
                color_path, source_folder_path
            )
        )

    if segmentation_path not in paths and segmentation_path is not None:
        logging.warning(
            "Specified segmentation_file file {} not in source path {}!".format(
                segmentation_path, segmentation_path
            )
        )

    logging.info("Segmentation file will also use uint8 as a datatype.")

    for path in paths:
        if path == color_path:
            convert_nifti(path, target_path, "color", "uint8", scale)
        elif path == segmentation_path:
            convert_nifti(path, target_path, "segmentation", "uint8", scale)
        else:
            convert_nifti(path, target_path, path.stem, "uint8", scale)


if __name__ == "__main__":
    args = create_parser().parse_args()
    setup_logging(args)

    source_path = Path(args.source_path)

    if source_path.is_dir():
        convert_folder_nifti(
            source_path,
            Path(args.target_path),
            source_path / args.color_file,
            source_path / args.segmentation_file,
            args.scale,
        )
    else:
        convert_nifti(
            source_path, Path(args.target_path), args.layer_name, args.dtype, args.scale
        )
