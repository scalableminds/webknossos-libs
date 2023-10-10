"""This module converts a NIFTI image or a folder with NIFTI images to a WEBKNOSSOS dataset."""

import logging
from pathlib import Path
from typing import Any, Optional, Tuple, Union, cast

import nibabel as nib
import numpy as np
import typer
from sklearn.preprocessing import LabelEncoder
from typing_extensions import Annotated

from ..dataset import DataFormat, Dataset, LayerCategoryType
from ..dataset.defaults import DEFAULT_CHUNK_SHAPE, DEFAULT_CHUNKS_PER_SHARD
from ..geometry import BoundingBox, Vec3Int
from ..utils import time_start, time_stop
from ._utils import (
    Vec2Int,
    VoxelSize,
    pad_or_crop_to_size_and_topleft,
    parse_bbox,
    parse_path,
    parse_vec3int,
    parse_voxel_size,
)


def to_target_datatype(
    data: np.ndarray,
    target_dtype: Union[type, str, np.dtype],
    is_segmentation_layer: bool,
) -> np.ndarray:
    """Cast data into target datatype."""

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
    voxel_size: Optional[VoxelSize],
    data_format: DataFormat,
    chunk_shape: Vec3Int,
    chunks_per_shard: Vec3Int,
    is_segmentation_layer: bool = False,
    bbox_to_enforce: Optional[BoundingBox] = None,
    use_orientation_header: bool = False,
    flip_axes: Optional[Union[int, Tuple[int, ...]]] = None,
) -> None:
    """Converts a single NIFTI file into a WEBKNOSSOS dataset."""

    shard_shape = chunk_shape * chunks_per_shard
    time_start(f"Converting of {source_nifti_path}")

    source_nifti = nib.load(str(source_nifti_path.resolve()))

    if use_orientation_header:
        # Get canonical representation of data to incorporate
        # encoded transformations. Needs to be flipped later
        # to match the coordinate system of WKW.
        source_nifti = nib.funcs.as_closest_canonical(source_nifti, enforce_diag=False)

    cube_data = np.array(source_nifti.get_fdata())  # type:ignore

    category_type: LayerCategoryType = (
        "segmentation" if is_segmentation_layer else "color"
    )
    logging.debug("Assuming %s as layer type for %s", category_type, layer_name)

    if len(source_nifti.shape) == 3:  # type:ignore
        cube_data = cube_data.reshape((1,) + source_nifti.shape)  # type:ignore

    elif len(source_nifti.shape) == 4:  # type:ignore
        cube_data = np.transpose(cube_data, (3, 0, 1, 2))

    else:
        logging.warning(
            "Converting of %s failed! Too many or too less dimensions",
            source_nifti_path,
        )

        return

    if use_orientation_header:
        # Flip y and z to transform data into wkw's coordinate system.
        cube_data = np.flip(cube_data, (2, 3))

    if flip_axes:
        cube_data = np.flip(cube_data, flip_axes)

    if voxel_size is None:
        voxel_size = tuple(map(float, source_nifti.header["pixdim"][:3]))  # type:ignore

    logging.info("Using voxel_size: %s", voxel_size)
    cube_data = to_target_datatype(cube_data, dtype, is_segmentation_layer)

    # everything needs to be padded to
    if bbox_to_enforce is not None:
        target_topleft = np.array((0,) + tuple(bbox_to_enforce.topleft))
        target_size = np.array((1,) + tuple(bbox_to_enforce.size))

        cube_data = pad_or_crop_to_size_and_topleft(
            cube_data, target_size, target_topleft
        )

    # Writing wkw compressed requires files of shape (shard_shape, shard_shape, shard_shape)
    # Pad data accordingly
    padding_offset = shard_shape - np.array(cube_data.shape[1:4]) % shard_shape
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
        "1", chunk_shape=chunk_shape, chunks_per_shard=chunks_per_shard
    )
    wk_mag.write(cube_data)

    time_stop(f"Converting of {source_nifti_path}")


def convert_folder_nifti(
    source_folder_path: Path,
    target_path: Path,
    color_subpath: Optional[str],
    segmentation_subpath: Optional[str],
    voxel_size: Optional[VoxelSize],
    data_format: DataFormat,
    chunk_shape: Vec3Int,
    chunks_per_shard: Vec3Int,
    use_orientation_header: bool = False,
    bbox_to_enforce: Optional[BoundingBox] = None,
    flip_axes: Optional[Union[int, Tuple[int, ...]]] = None,
) -> None:
    """Converts a folder of NIFTI files into WEBKNOSSOS dataset."""
    paths = list(source_folder_path.rglob("**/*.nii"))

    color_path = None
    segmentation_path = None
    if color_subpath is not None:
        color_path = target_path / color_subpath
        if color_path not in paths:
            logging.warning(
                "Specified color file %s not in source path %s!",
                color_path,
                source_folder_path,
            )

    if segmentation_subpath is not None:
        segmentation_path = target_path / segmentation_subpath
        if segmentation_path not in paths:
            logging.warning(
                "Specified segmentation_file file %s not in source path %s!",
                segmentation_path,
                source_folder_path,
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
                chunk_shape,
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
                chunk_shape,
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
                chunk_shape,
                chunks_per_shard,
                is_segmentation_layer=False,
                bbox_to_enforce=bbox_to_enforce,
                use_orientation_header=use_orientation_header,
                flip_axes=flip_axes,
            )


def main(
    *,
    source: Annotated[
        Any,
        typer.Argument(
            help="Path to your image data.",
            show_default=False,
            parser=parse_path,
        ),
    ],
    target: Annotated[
        Any,
        typer.Argument(
            help="Target path to save your WEBKNOSSOS dataset.",
            show_default=False,
            parser=parse_path,
        ),
    ],
    layer_name: Annotated[
        str,
        typer.Option(help="Name of the cubed layer (color or segmentation)"),
    ] = "color",
    voxel_size: Annotated[
        Optional[VoxelSize],
        typer.Option(
            help="The size of one voxel in source data in nanometers. \
Should be a comma seperated string (e.g. 11.0,11.0,20.0).",
            parser=parse_voxel_size,
            metavar="VOXEL_SIZE",
        ),
    ] = None,
    dtype: Annotated[
        str, typer.Option(help="Target datatype (e.g. uint8, uint16, uint32)")
    ] = "uint8",
    data_format: Annotated[
        DataFormat,
        typer.Option(
            help="Data format to store the target dataset in.",
        ),
    ] = "wkw",  # type:ignore
    chunk_shape: Annotated[
        Vec3Int,
        typer.Option(
            help="Number of voxels to be stored as a chunk in the output format "
            "(e.g. `32` or `32,32,32`).",
            parser=parse_vec3int,
            metavar="Vec3Int",
        ),
    ] = DEFAULT_CHUNK_SHAPE,
    chunks_per_shard: Annotated[
        Vec3Int,
        typer.Option(
            help="Number of chunks to be stored as a shard in the output format "
            "(e.g. `32` or `32,32,32`).",
            parser=parse_vec3int,
            metavar="Vec3Int",
        ),
    ] = DEFAULT_CHUNKS_PER_SHARD,
    enforce_bounding_box: Annotated[
        Optional[BoundingBox],
        typer.Option(
            help="The BoundingBox to which the input data should be written. "
            "If the input data is too small, it will be padded. If it's too large, "
            "it will be cropped. The input format is x,y,z,width,height,depth.",
            parser=parse_bbox,
            metavar="BBOX",
        ),
    ] = None,
    color_file: Annotated[
        Optional[str],
        typer.Option(
            help="When converting folder, name of file to become color layer."
        ),
    ] = None,
    segmentation_file: Annotated[
        Optional[str],
        typer.Option(
            help="When converting folder, name of file to become segmentation layer."
        ),
    ] = None,
    use_orientation_header: Annotated[
        bool,
        typer.Option(
            help="Use orientation information from header to interpret the input data \
(should be tried if output orientation seems to be wrong)."
        ),
    ] = False,
    is_segmentation_layer: Annotated[
        bool,
        typer.Option(
            help="When converting one layer, signals whether layer is segmentation layer. \
When converting a folder, this option is ignored."
        ),
    ] = False,
    flip_axes: Annotated[
        Optional[Vec2Int],
        typer.Option(
            help="The axes at which should be flipped. \
Input format is a comma separated list of axis indices. \
For example, 1,2,3 will flip the x, y and z axes.",
            parser=parse_vec3int,
            metavar="Vec3Int",
        ),
    ] = None,
) -> None:
    """Converts a NIFTI file or a folder of NIFTI files into a WEBKNOSSOS dataset."""

    if flip_axes is not None:
        for index in flip_axes:
            assert (
                0 <= index <= 3
            ), "flip_axes parameter must only contain indices between 0 and 3."

    if source.is_dir():
        convert_folder_nifti(
            source,
            target,
            color_file,
            segmentation_file,
            voxel_size=voxel_size,
            data_format=data_format,
            chunk_shape=chunk_shape,
            chunks_per_shard=chunks_per_shard,
            bbox_to_enforce=enforce_bounding_box,
            use_orientation_header=use_orientation_header,
            flip_axes=flip_axes,
        )
    else:
        convert_nifti(
            source,
            target,
            layer_name,
            dtype,
            voxel_size=voxel_size,
            data_format=data_format,
            chunk_shape=chunk_shape,
            chunks_per_shard=chunks_per_shard,
            is_segmentation_layer=is_segmentation_layer,
            bbox_to_enforce=enforce_bounding_box,
            use_orientation_header=use_orientation_header,
            flip_axes=flip_axes,
        )
