import logging
import time
from argparse import ArgumentParser
from pathlib import Path

import nibabel as nib
import numpy as np

from wkcuber.api.Dataset import TiffDataset, WKDataset
from wkcuber.utils import (
    DEFAULT_WKW_FILE_LEN,
    DEFAULT_WKW_VOXELS_PER_BLOCK,
    add_scale_flag,
    add_verbose_flag,
    pad_or_crop_to_size_and_topleft,
    parse_bounding_box,
    setup_logging,
)


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

    parser.add_argument(
        "--write_tiff",
        help="Output tiff dataset instead of wkw.",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--use_orientation_header",
        help="Use orientation information from header to interpret the input data (should be tried if output orientation seems to be wrong).",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--enforce_bounding_box",
        help="The BoundingBox to which the input data should be written. If the input data is too small, it will be padded. If it's too large, it will be cropped. The input format is x,y,z,width,height,depth.",
        default=None,
        type=parse_bounding_box,
    )

    parser.add_argument(
        "--flip_axes",
        help="The axes at which should be flipped. Input format is a comma separated list of axis indices. For example, 1,2,3 will flip the x, y and z axes.",
        default=None,
    )

    add_scale_flag(parser, required=False)
    add_verbose_flag(parser)

    return parser


def to_target_datatype(
    data: np.ndarray, target_dtype, is_probably_binary: bool
) -> np.ndarray:
    if is_probably_binary:
        logging.info(
            f"Casting directly to {target_dtype}, as input seems to be binary."
        )
        return data.astype(np.dtype(target_dtype))

    if data.dtype == np.dtype("float32"):
        factor = data.max()
    elif data.dtype == np.dtype("float64"):
        factor = data.max() / np.iinfo(target_dtype).max
    else:
        factor = np.iinfo(data.dtype).max / np.iinfo(target_dtype).max

    if data.max() == 0:
        logging.warning("Not rescaling data since maximum is 0")
        factor = 1

    return (data / factor).astype(np.dtype(target_dtype))


def convert_nifti(
    source_nifti_path,
    target_path,
    layer_name,
    dtype,
    scale,
    mag=1,
    file_len=DEFAULT_WKW_FILE_LEN,
    bbox_to_enforce=None,
    write_tiff=False,
    use_orientation_header=False,
    flip_axes=None,
):
    voxels_per_cube = file_len * DEFAULT_WKW_VOXELS_PER_BLOCK
    ref_time = time.time()

    source_nifti = nib.load(str(source_nifti_path.resolve()))

    if use_orientation_header:
        # Get canonical representation of data to incorporate
        # encoded transformations. Needs to be flipped later
        # to match the coordinate system of WKW.
        source_nifti = nib.funcs.as_closest_canonical(source_nifti, enforce_diag=False)

    cube_data = np.array(source_nifti.get_fdata())

    is_probably_binary = np.unique(cube_data).shape[0] == 2
    assume_segmentation_layer = (
        False
    )  # Since webKnossos does not support multiple segmention layers, this is hardcoded to False right now.
    max_cell_id_args = (
        {"largest_segment_id": int(np.max(cube_data) + 1)}
        if assume_segmentation_layer
        else {}
    )
    category_type = "segmentation" if assume_segmentation_layer else "color"
    logging.debug(f"Assuming {category_type} as layer type for {layer_name}")

    if len(source_nifti.shape) == 3:
        cube_data = cube_data.reshape((1,) + source_nifti.shape)

    elif len(source_nifti.shape) == 4:
        cube_data = np.transpose(cube_data, (3, 0, 1, 2))

    else:
        logging.warning(
            "Converting of {} failed! Too many or too less dimensions".format(
                source_nifti_path
            )
        )

        return

    if use_orientation_header:
        # Flip y and z to transform data into wkw's coordinate system.
        cube_data = np.flip(cube_data, (2, 3))

    if flip_axes:
        cube_data = np.flip(cube_data, flip_axes)

    if scale is None:
        scale = tuple(map(float, source_nifti.header["pixdim"][:3]))
    logging.info(f"Using scale: {scale}")
    cube_data = to_target_datatype(cube_data, dtype, is_probably_binary)

    # everything needs to be padded to
    if bbox_to_enforce is not None:
        target_topleft = np.array((0,) + tuple(bbox_to_enforce.topleft))
        target_size = np.array((1,) + tuple(bbox_to_enforce.size))

        cube_data = pad_or_crop_to_size_and_topleft(
            cube_data, target_size, target_topleft
        )

    # Writing wkw compressed requires files of shape (voxels_per_cube, voxels_per_cube, voxels_per_cube)
    # Pad data accordingly
    padding_offset = voxels_per_cube - np.array(cube_data.shape[1:4]) % voxels_per_cube
    padding_offset = (0, 0, 0)
    cube_data = np.pad(
        cube_data,
        (
            (0, 0),
            (0, int(padding_offset[0])),
            (0, int(padding_offset[1])),
            (0, int(padding_offset[2])),
        ),
    )

    if write_tiff:
        ds = TiffDataset.get_or_create(target_path, scale=scale or (1, 1, 1))
        layer = ds.get_or_add_layer(
            layer_name,
            category_type,
            dtype_per_layer=np.dtype(dtype),
            **max_cell_id_args,
        )
        mag = layer.get_or_add_mag("1")

        mag.write(cube_data.squeeze())
    else:
        ds = WKDataset.get_or_create(target_path, scale=scale or (1, 1, 1))
        layer = ds.get_or_add_layer(
            layer_name,
            category_type,
            dtype_per_layer=np.dtype(dtype),
            **max_cell_id_args,
        )
        mag = layer.get_or_add_mag("1", file_len=file_len)
        mag.write(cube_data)

    logging.debug(
        "Converting of {} took {:.8f}s".format(
            source_nifti_path, time.time() - ref_time
        )
    )


def convert_folder_nifti(
    source_folder_path,
    target_path,
    color_subpath,
    segmentation_subpath,
    scale,
    use_orientation_header=False,
    bbox_to_enforce=None,
    write_tiff=False,
    flip_axes=None,
):
    paths = list(source_folder_path.rglob("**/*.nii"))

    color_path = None
    segmentation_path = None
    if color_subpath is not None:
        color_path = target_path / color_subpath
        if color_path not in paths:
            logging.warning(
                "Specified color file {} not in source path {}!".format(
                    color_path, source_folder_path
                )
            )

    if segmentation_subpath is not None:
        segmentation_path = target_path / segmentation_subpath
        if segmentation_path not in paths:
            logging.warning(
                "Specified segmentation_file file {} not in source path {}!".format(
                    segmentation_path, segmentation_path
                )
            )

    logging.info("Segmentation file will also use uint8 as a datatype.")

    conversion_args = {
        "scale": scale,
        "write_tiff": write_tiff,
        "bbox_to_enforce": bbox_to_enforce,
        "use_orientation_header": use_orientation_header,
        "flip_axes": flip_axes,
    }
    for path in paths:
        if path == color_path:
            convert_nifti(path, target_path, "color", "uint8", **conversion_args)
        elif path == segmentation_path:
            convert_nifti(path, target_path, "segmentation", "uint8", **conversion_args)
        else:
            convert_nifti(path, target_path, path.stem, "uint8", **conversion_args)


def main():
    args = create_parser().parse_args()
    setup_logging(args)

    source_path = Path(args.source_path)

    flip_axes = None
    if args.flip_axes is not None:
        flip_axes = tuple(int(x) for x in args.flip_axes.split(","))
        for index in flip_axes:
            assert (
                0 <= index <= 3
            ), "flip_axes parameter must only contain indices between 0 and 3."

    conversion_args = {
        "scale": args.scale,
        "write_tiff": args.write_tiff,
        "bbox_to_enforce": args.enforce_bounding_box,
        "use_orientation_header": args.use_orientation_header,
        "flip_axes": flip_axes,
    }

    if source_path.is_dir():
        convert_folder_nifti(
            source_path,
            Path(args.target_path),
            args.color_file,
            args.segmentation_file,
            **conversion_args,
        )
    else:
        convert_nifti(
            source_path,
            Path(args.target_path),
            args.layer_name,
            args.dtype,
            **conversion_args,
        )


if __name__ == "__main__":
    main()
