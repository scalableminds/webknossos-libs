import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Optional, Tuple, Union, cast

import nibabel as nib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from webknossos import BoundingBox, DataFormat, Dataset, LayerCategoryType, Vec3Int
from webknossos.utils import time_start, time_stop

from ._internal.utils import (
    add_data_format_flags,
    add_voxel_size_flag,
    add_verbose_flag,
    pad_or_crop_to_size_and_topleft,
    parse_bounding_box,
    parse_path,
    setup_logging,
    setup_warnings,
)


def create_parser() -> ArgumentParser:
    parser = ArgumentParser()

    parser.add_argument(
        "source_path",
        help="Path to NIFTY file or to a directory if multiple NIFTI files should be converted. "
        "In the latter case, also see --color_file and --segmentation_file.",
        type=Path,
    )

    parser.add_argument(
        "target_path",
        help="Output directory for the generated WKW dataset.",
        type=parse_path,
    )

    parser.add_argument(
        "--layer_name",
        "-l",
        help="Name of the cubed layer (color or segmentation).",
        default="color",
    )

    parser.add_argument(
        "--is_segmentation_layer",
        "-sl",
        help="When converting one layer, signals whether layer is segmentation layer. "
        "When converting a folder, this option is ignored",
        default=False,
        action="store_true",
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

    add_voxel_size_flag(parser, required=False)
    add_verbose_flag(parser)
    add_data_format_flags(parser)

    return parser


def to_target_datatype(
    data: np.ndarray,
    target_dtype: Union[type, str, np.dtype],
    is_segmentation_layer: bool,
) -> np.ndarray:
    if is_segmentation_layer:
        original_shape = data.shape
        label_encoder = LabelEncoder()
        return (
            label_encoder.fit_transform(data.ravel())
            .reshape(original_shape)
            .astype(np.dtype(target_dtype))
        )

    factor: Any
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
    source_nifti_path: Path,
    target_path: Path,
    layer_name: str,
    dtype: str,
    voxel_size: Tuple[float, ...],
    data_format: DataFormat,
    chunk_size: Vec3Int,
    chunks_per_shard: Vec3Int,
    is_segmentation_layer: bool = False,
    bbox_to_enforce: Optional[BoundingBox] = None,
    use_orientation_header: bool = False,
    flip_axes: Optional[Union[int, Tuple[int, ...]]] = None,
) -> None:
    shard_size = chunk_size * chunks_per_shard
    time_start(f"Converting of {source_nifti_path}")

    source_nifti = nib.load(str(source_nifti_path.resolve()))

    if use_orientation_header:
        # Get canonical representation of data to incorporate
        # encoded transformations. Needs to be flipped later
        # to match the coordinate system of WKW.
        source_nifti = nib.funcs.as_closest_canonical(source_nifti, enforce_diag=False)

    cube_data = np.array(source_nifti.get_fdata())

    category_type: LayerCategoryType = (
        "segmentation" if is_segmentation_layer else "color"
    )
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

    if voxel_size is None:
        voxel_size = tuple(map(float, source_nifti.header["pixdim"][:3]))

    logging.info(f"Using voxel_size: {voxel_size}")
    cube_data = to_target_datatype(cube_data, dtype, is_segmentation_layer)

    # everything needs to be padded to
    if bbox_to_enforce is not None:
        target_topleft = np.array((0,) + tuple(bbox_to_enforce.topleft))
        target_size = np.array((1,) + tuple(bbox_to_enforce.size))

        cube_data = pad_or_crop_to_size_and_topleft(
            cube_data, target_size, target_topleft
        )

    # Writing wkw compressed requires files of shape (shard_size, shard_size, shard_size)
    # Pad data accordingly
    padding_offset = shard_size - np.array(cube_data.shape[1:4]) % shard_size
    cube_data = np.pad(
        cube_data,
        (
            (0, 0),
            (0, int(padding_offset[0])),
            (0, int(padding_offset[1])),
            (0, int(padding_offset[2])),
        ),
    )

    wk_ds = Dataset(
        target_path,
        voxel_size=cast(Tuple[float, float, float], voxel_size or (1, 1, 1)),
        exist_ok=True,
    )
    wk_layer = (
        wk_ds.get_or_add_layer(
            layer_name,
            category_type,
            dtype_per_layer=np.dtype(dtype),
            data_format=data_format,
            largest_segment_id=int(np.max(cube_data) + 1),
        )
        if is_segmentation_layer
        else wk_ds.get_or_add_layer(
            layer_name,
            category_type,
            data_format=data_format,
            dtype_per_layer=np.dtype(dtype),
        )
    )
    wk_mag = wk_layer.get_or_add_mag(
        "1", chunk_size=chunk_size, chunks_per_shard=chunks_per_shard
    )
    wk_mag.write(cube_data)

    time_stop(f"Converting of {source_nifti_path}")


def convert_folder_nifti(
    source_folder_path: Path,
    target_path: Path,
    color_subpath: str,
    segmentation_subpath: str,
    voxel_size: Tuple[float, ...],
    data_format: DataFormat,
    chunk_size: Vec3Int,
    chunks_per_shard: Vec3Int,
    use_orientation_header: bool = False,
    bbox_to_enforce: Optional[BoundingBox] = None,
    flip_axes: Optional[Union[int, Tuple[int, ...]]] = None,
) -> None:
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

    for path in paths:
        if path == color_path:
            convert_nifti(
                path,
                target_path,
                "color",
                "uint8",
                voxel_size,
                data_format,
                chunk_size,
                chunks_per_shard,
                is_segmentation_layer=False,
                bbox_to_enforce=bbox_to_enforce,
                use_orientation_header=use_orientation_header,
                flip_axes=flip_axes,
            )
        elif path == segmentation_path:
            convert_nifti(
                path,
                target_path,
                "segmentation",
                "uint8",
                voxel_size,
                data_format,
                chunk_size,
                chunks_per_shard,
                is_segmentation_layer=True,
                bbox_to_enforce=bbox_to_enforce,
                use_orientation_header=use_orientation_header,
                flip_axes=flip_axes,
            )
        else:
            convert_nifti(
                path,
                target_path,
                path.stem,
                "uint8",
                voxel_size,
                data_format,
                chunk_size,
                chunks_per_shard,
                is_segmentation_layer=False,
                bbox_to_enforce=bbox_to_enforce,
                use_orientation_header=use_orientation_header,
                flip_axes=flip_axes,
            )


def main(args: Namespace) -> None:
    source_path = args.source_path

    flip_axes = None
    if args.flip_axes is not None:
        flip_axes = tuple(int(x) for x in args.flip_axes.split(","))
        for index in flip_axes:
            assert (
                0 <= index <= 3
            ), "flip_axes parameter must only contain indices between 0 and 3."

    if source_path.is_dir():
        convert_folder_nifti(
            source_path,
            args.target_path,
            args.color_file,
            args.segmentation_file,
            voxel_size=args.voxel_size,
            data_format=args.data_format,
            chunk_size=args.chunk_size,
            chunks_per_shard=args.chunks_per_shard,
            bbox_to_enforce=args.enforce_bounding_box,
            use_orientation_header=args.use_orientation_header,
            flip_axes=flip_axes,
        )
    else:
        convert_nifti(
            source_path,
            args.target_path,
            args.layer_name,
            args.dtype,
            voxel_size=args.voxel_size,
            data_format=args.data_format,
            chunk_size=args.chunk_size,
            chunks_per_shard=args.chunks_per_shard,
            is_segmentation_layer=args.is_segmentation_layer,
            bbox_to_enforce=args.enforce_bounding_box,
            use_orientation_header=args.use_orientation_header,
            flip_axes=flip_axes,
        )


if __name__ == "__main__":
    setup_warnings()
    args = create_parser().parse_args()
    setup_logging(args)

    main(args)
