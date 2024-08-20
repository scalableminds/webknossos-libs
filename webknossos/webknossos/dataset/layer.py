import logging
import operator
import re
import warnings
from argparse import Namespace
from os import PathLike
from os.path import relpath
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np
from cluster_tools import Executor
from numpy.typing import DTypeLike
from upath import UPath

from ..geometry import Mag, NDBoundingBox, Vec3Int, Vec3IntLike
from ._array import ArrayException, BaseArray, DataFormat
from ._downsampling_utils import (
    calculate_default_coarsest_mag,
    calculate_mags_to_downsample,
    calculate_mags_to_upsample,
    determine_buffer_shape,
    downsample_cube_job,
    parse_interpolation_mode,
)
from ._upsampling_utils import upsample_cube_job
from .layer_categories import COLOR_CATEGORY, SEGMENTATION_CATEGORY, LayerCategoryType
from .properties import (
    LayerProperties,
    LayerViewConfiguration,
    MagViewProperties,
    SegmentationLayerProperties,
    _properties_floating_type_to_python_type,
    _python_floating_type_to_properties_type,
)
from .sampling_modes import SamplingModes
from .view import View, _copy_job

if TYPE_CHECKING:
    from .dataset import Dataset

from ..utils import (
    copytree,
    get_executor_for_args,
    is_fs_path,
    named_partial,
    rmtree,
    warn_deprecated,
)
from .defaults import (
    DEFAULT_CHUNK_SHAPE,
    DEFAULT_CHUNKS_PER_SHARD,
    DEFAULT_CHUNKS_PER_SHARD_ZARR,
)
from .mag_view import MagView, _find_mag_path_on_disk


def _is_int(s: str) -> bool:
    try:
        int(s)
        return True
    except ValueError:
        return False


def _convert_dtypes(
    dtype: DTypeLike,
    num_channels: int,
    dtype_per_layer_to_dtype_per_channel: bool,
) -> str:
    op = operator.truediv if dtype_per_layer_to_dtype_per_channel else operator.mul

    # split the dtype into the actual type and the number of bits
    # example: "uint24" -> ["uint", "24"]
    dtype_parts = re.split(r"(\d+)", str(dtype))
    # calculate number of bits for dtype_per_channel
    converted_dtype_parts = [
        (str(int(op(int(part), num_channels))) if _is_int(part) else part)
        for part in dtype_parts
    ]
    return "".join(converted_dtype_parts)


def _normalize_dtype_per_channel(dtype_per_channel: DTypeLike) -> np.dtype:
    try:
        return np.dtype(dtype_per_channel)
    except TypeError as e:
        raise TypeError(
            "Cannot add layer. The specified 'dtype_per_channel' must be a valid dtype."
        ) from e


def _normalize_dtype_per_layer(dtype_per_layer: DTypeLike) -> DTypeLike:
    try:
        dtype_per_layer = str(np.dtype(dtype_per_layer))
    except Exception:
        pass  # casting to np.dtype fails if the user specifies a special dtype like "uint24"
    return dtype_per_layer  # type: ignore[return-value]


def _dtype_per_layer_to_dtype_per_channel(
    dtype_per_layer: DTypeLike, num_channels: int
) -> np.dtype:
    try:
        return np.dtype(
            _convert_dtypes(
                dtype_per_layer, num_channels, dtype_per_layer_to_dtype_per_channel=True
            )
        )
    except TypeError as e:
        raise TypeError(
            "Converting dtype_per_layer to dtype_per_channel failed. Double check if the dtype_per_layer value is correct."
        ) from e


def _dtype_per_channel_to_dtype_per_layer(
    dtype_per_channel: DTypeLike, num_channels: int
) -> str:
    return _convert_dtypes(
        np.dtype(dtype_per_channel),
        num_channels,
        dtype_per_layer_to_dtype_per_channel=False,
    )


def _dtype_per_channel_to_element_class(
    dtype_per_channel: DTypeLike, num_channels: int
) -> str:
    dtype_per_layer = _dtype_per_channel_to_dtype_per_layer(
        dtype_per_channel, num_channels
    )
    return _python_floating_type_to_properties_type.get(
        dtype_per_layer, dtype_per_layer
    )


def _element_class_to_dtype_per_channel(
    element_class: str, num_channels: int
) -> np.dtype:
    dtype_per_layer = _properties_floating_type_to_python_type.get(
        element_class, element_class
    )
    return _dtype_per_layer_to_dtype_per_channel(dtype_per_layer, num_channels)


def _get_sharding_parameters(
    *,
    chunk_shape: Optional[Union[Vec3IntLike, int]],
    chunks_per_shard: Optional[Union[Vec3IntLike, int]],
    chunk_size: Optional[Union[Vec3IntLike, int]],  # deprecated
    block_len: Optional[int],  # deprecated
    file_len: Optional[int],  # deprecated
) -> Tuple[Optional[Vec3Int], Optional[Vec3Int]]:
    if chunk_shape is not None:
        chunk_shape = Vec3Int.from_vec_or_int(chunk_shape)
    elif chunk_size is not None:
        warn_deprecated("chunk_size", "chunk_shape")
        chunk_shape = Vec3Int.from_vec_or_int(chunk_size)
    elif block_len is not None:
        warn_deprecated("block_len", "chunk_shape")
        chunk_shape = Vec3Int.full(block_len)

    if chunks_per_shard is not None:
        chunks_per_shard = Vec3Int.from_vec_or_int(chunks_per_shard)
    elif file_len is not None:
        warn_deprecated("file_len", "chunks_per_shard")
        chunks_per_shard = Vec3Int.full(file_len)

    return (chunk_shape, chunks_per_shard)


class Layer:
    """
    A `Layer` consists of multiple `MagView`s, which store the same data in different magnifications.
    """

    def __init__(self, dataset: "Dataset", properties: LayerProperties) -> None:
        """
        Do not use this constructor manually. Instead use `Dataset.add_layer()` to create a `Layer`.
        """
        # It is possible that the properties on disk do not contain the number of channels.
        # Therefore, the parameter is optional. However at this point, 'num_channels' was already inferred.
        assert properties.num_channels is not None

        self._name: str = properties.name  # The name is also stored in the properties, but the name is required to get the properties.
        self._dataset = dataset
        self._dtype_per_channel = _element_class_to_dtype_per_channel(
            properties.element_class, properties.num_channels
        )
        self._mags: Dict[Mag, MagView] = {}

        self.path.mkdir(parents=True, exist_ok=True)

        for mag in properties.mags:
            self._setup_mag(Mag(mag.mag), mag.path)
        # Only keep the properties of mags that were initialized.
        # Sometimes the directory of a mag is removed from disk manually, but the properties are not updated.
        self._properties.mags = [
            res for res in self._properties.mags if Mag(res.mag) in self._mags
        ]

    @property
    def path(self) -> Path:
        return self.dataset.path / self.name

    @property
    def _properties(self) -> LayerProperties:
        return next(
            layer_property
            for layer_property in self.dataset._properties.data_layers
            if layer_property.name == self.name
        )

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, layer_name: str) -> None:
        """
        Renames the layer to `layer_name`. This changes the name of the directory on disk and updates the properties.
        Only layers on local file systems can be renamed.
        """
        if layer_name == self.name:
            return
        self.dataset._ensure_writable()
        assert (
            layer_name not in self.dataset.layers.keys()
        ), f"Failed to rename layer {self.name} to {layer_name}: The new name already exists."
        assert is_fs_path(self.path), f"Cannot rename remote layer {self.path}"
        assert (
            "/" not in layer_name
        ), f"Cannot rename layer, because there is a '/' character in the layer name: {layer_name}"
        self.path.rename(self.dataset.path / layer_name)
        del self.dataset.layers[self.name]
        self.dataset.layers[layer_name] = self
        self._properties.name = layer_name
        self._name = layer_name

        # The MagViews need to be updated
        for mag in self._mags.values():
            mag._path = _find_mag_path_on_disk(self.dataset.path, self.name, mag.name)
            # Deleting the dataset will close the file handle.
            # The new dataset will be opened automatically when needed.
            del mag._array

        self.dataset._export_as_json()

    @property
    def dataset(self) -> "Dataset":
        return self._dataset

    @property
    def bounding_box(self) -> NDBoundingBox:
        return self._properties.bounding_box

    @bounding_box.setter
    def bounding_box(self, bbox: NDBoundingBox) -> None:
        """
        Updates the offset and size of the bounding box of this layer in the properties.
        """
        self.dataset._ensure_writable()
        assert bbox.topleft.is_positive(), f"Updating the bounding box of layer {self} to {bbox} failed, topleft must not contain negative dimensions."
        self._properties.bounding_box = bbox
        self.dataset._export_as_json()
        for mag in self.mags.values():
            mag._array.ensure_size(bbox.align_with_mag(mag.mag).in_mag(mag.mag))

    @property
    def category(self) -> LayerCategoryType:
        return COLOR_CATEGORY

    @property
    def dtype_per_layer(self) -> str:
        return _dtype_per_channel_to_dtype_per_layer(
            self.dtype_per_channel, self.num_channels
        )

    @property
    def dtype_per_channel(self) -> np.dtype:
        return self._dtype_per_channel

    @property
    def num_channels(self) -> int:
        assert self._properties.num_channels is not None
        return self._properties.num_channels

    @property
    def data_format(self) -> DataFormat:
        assert self._properties.data_format is not None
        return self._properties.data_format

    @property
    def default_view_configuration(self) -> Optional[LayerViewConfiguration]:
        return self._properties.default_view_configuration

    @default_view_configuration.setter
    def default_view_configuration(
        self, view_configuration: LayerViewConfiguration
    ) -> None:
        self.dataset._ensure_writable()
        self._properties.default_view_configuration = view_configuration
        self.dataset._export_as_json()  # update properties on disk

    @property
    def read_only(self) -> bool:
        return self.dataset.read_only

    @property
    def mags(self) -> Dict[Mag, MagView]:
        """
        Getter for dictionary containing all mags.
        """
        return self._mags

    def get_mag(self, mag: Union[int, str, list, tuple, np.ndarray, Mag]) -> MagView:
        """
        Returns the `MagView` called `mag` of this layer. The return type is `webknossos.dataset.mag_view.MagView`.

        This function raises an `IndexError` if the specified `mag` does not exist.
        """
        mag = Mag(mag)
        if mag not in self.mags.keys():
            raise IndexError(
                "The mag {} is not a mag of this layer".format(mag.to_layer_name())
            )
        return self.mags[mag]

    def get_finest_mag(self) -> MagView:
        return self.get_mag(min(self.mags.keys()))

    def get_best_mag(self) -> MagView:
        """Deprecated, please use `get_finest_mag`."""
        warn_deprecated("get_best_mag()", "get_finest_mag()")
        return self.get_finest_mag()

    def add_mag(
        self,
        mag: Union[int, str, list, tuple, np.ndarray, Mag],
        chunk_shape: Optional[Union[Vec3IntLike, int]] = None,  # DEFAULT_CHUNK_SHAPE,
        chunks_per_shard: Optional[
            Union[int, Vec3IntLike]
        ] = None,  # DEFAULT_CHUNKS_PER_SHARD,
        compress: bool = False,
        *,
        chunk_size: Optional[Union[Vec3IntLike, int]] = None,  # deprecated
        block_len: Optional[int] = None,  # deprecated
        file_len: Optional[int] = None,  # deprecated
    ) -> MagView:
        """
        Creates a new mag called and adds it to the layer.
        The parameter `chunk_shape`, `chunks_per_shard` and `compress` can be
        specified to adjust how the data is stored on disk.
        Note that writing compressed data which is not aligned with the blocks on disk may result in
        diminished performance, as full blocks will automatically be read to pad the write actions. Alternatively,
        you can call mag.compress() after all the data was written

        The return type is `webknossos.dataset.mag_view.MagView`.

        Raises an IndexError if the specified `mag` already exists.
        """
        self.dataset._ensure_writable()
        # normalize the name of the mag
        mag = Mag(mag)
        compression_mode = compress

        chunk_shape, chunks_per_shard = _get_sharding_parameters(
            chunk_shape=chunk_shape,
            chunks_per_shard=chunks_per_shard,
            chunk_size=chunk_size,
            block_len=block_len,
            file_len=file_len,
        )
        if chunk_shape is None:
            chunk_shape = DEFAULT_CHUNK_SHAPE
        if chunks_per_shard is None:
            if self.data_format == DataFormat.Zarr:
                chunks_per_shard = DEFAULT_CHUNKS_PER_SHARD_ZARR
            else:
                chunks_per_shard = DEFAULT_CHUNKS_PER_SHARD

        if chunk_shape not in (Vec3Int.full(32), Vec3Int.full(64)):
            warnings.warn(
                "[INFO] `chunk_shape` of `32, 32, 32` or `64, 64, 64` is recommended for optimal "
                + f"performance in WEBKNOSSOS. Got {chunk_shape}."
            )

        self._assert_mag_does_not_exist_yet(mag)
        self._create_dir_for_mag(mag)

        mag_view = MagView(
            self,
            mag,
            chunk_shape=chunk_shape,
            chunks_per_shard=chunks_per_shard,
            compression_mode=compression_mode,
            create=True,
        )

        mag_view._array.ensure_size(self.bounding_box.align_with_mag(mag).in_mag(mag))

        self._mags[mag] = mag_view
        mag_array_info = mag_view.info
        self._properties.mags += [
            MagViewProperties(
                mag=Mag(mag_view.name),
                cube_length=(
                    mag_array_info.shard_shape.x
                    if mag_array_info.data_format == DataFormat.WKW
                    else None
                ),
                axis_order=(
                    dict(
                        zip(
                            ("c", "x", "y", "z"),
                            (0, *self.bounding_box.index_xyz),
                        )
                    )
                    if mag_array_info.data_format in (DataFormat.Zarr, DataFormat.Zarr3)
                    else None
                ),
            )
        ]

        self.dataset._export_as_json()

        return self._mags[mag]

    def add_mag_for_existing_files(
        self,
        mag: Union[int, str, list, tuple, np.ndarray, Mag],
    ) -> MagView:
        """
        Creates a new mag based on already existing files.

        Raises an IndexError if the specified `mag` does not exists.
        """
        self.dataset._ensure_writable()
        mag = Mag(mag)
        assert (
            mag not in self.mags
        ), f"Cannot add mag {mag} as it already exists for layer {self}"
        self._setup_mag(mag)
        mag_view = self._mags[mag]
        mag_array_info = mag_view.info
        self._properties.mags.append(
            MagViewProperties(
                mag=mag,
                cube_length=(
                    mag_array_info.shard_shape.x
                    if mag_array_info.data_format == DataFormat.WKW
                    else None
                ),
                axis_order=(
                    {
                        key: value
                        for key, value in zip(
                            ("c", "x", "y", "z"),
                            (0, *self.bounding_box.index_xyz),
                        )
                    }
                    if mag_array_info.data_format in (DataFormat.Zarr, DataFormat.Zarr3)
                    else None
                ),
            )
        )
        self.dataset._export_as_json()

        return mag_view

    def get_or_add_mag(
        self,
        mag: Union[int, str, list, tuple, np.ndarray, Mag],
        chunk_shape: Optional[Union[Vec3IntLike, int]] = None,
        chunks_per_shard: Optional[Union[Vec3IntLike, int]] = None,
        compress: Optional[bool] = None,
        *,
        chunk_size: Optional[Union[Vec3IntLike, int]] = None,  # deprecated
        block_len: Optional[int] = None,  # deprecated
        file_len: Optional[int] = None,  # deprecated
    ) -> MagView:
        """
        Creates a new mag and adds it to the dataset, in case it did not exist before.
        Then, returns the mag.

        See `add_mag` for more information.
        """

        # normalize the name of the mag
        mag = Mag(mag)
        compression_mode = compress

        chunk_shape, chunks_per_shard = _get_sharding_parameters(
            chunk_shape=chunk_shape,
            chunks_per_shard=chunks_per_shard,
            chunk_size=chunk_size,
            block_len=block_len,
            file_len=file_len,
        )

        if mag in self._mags.keys():
            assert (
                chunk_shape is None or self._mags[mag].info.chunk_shape == chunk_shape
            ), f"Cannot get_or_add_mag: The mag {mag} already exists, but the chunk sizes do not match"
            assert (
                chunks_per_shard is None
                or self._mags[mag].info.chunks_per_shard == chunks_per_shard
            ), f"Cannot get_or_add_mag: The mag {mag} already exists, but the chunks per shard do not match"
            assert (
                compression_mode is None
                or self._mags[mag].info.compression_mode == compression_mode
            ), f"Cannot get_or_add_mag: The mag {mag} already exists, but the compression modes do not match"
            return self.get_mag(mag)
        else:
            return self.add_mag(
                mag,
                chunk_shape=chunk_shape,
                chunks_per_shard=chunks_per_shard,
                compress=compression_mode or False,
            )

    def delete_mag(self, mag: Union[int, str, list, tuple, np.ndarray, Mag]) -> None:
        """
        Deletes the MagView from the `datasource-properties.json` and the data from disk.

        This function raises an `IndexError` if the specified `mag` does not exist.
        """
        self.dataset._ensure_writable()
        mag = Mag(mag)
        if mag not in self.mags.keys():
            raise IndexError(
                "Deleting mag {} failed. There is no mag with this name".format(mag)
            )

        del self._mags[mag]
        self._properties.mags = [
            res for res in self._properties.mags if Mag(res.mag) != mag
        ]
        self.dataset._export_as_json()
        # delete files on disk
        full_path = _find_mag_path_on_disk(
            self.dataset.path, self.name, mag.to_layer_name()
        )
        rmtree(full_path)

    def add_copy_mag(
        self,
        foreign_mag_view_or_path: Union[PathLike, str, MagView],
        extend_layer_bounding_box: bool = True,
        chunk_shape: Optional[Union[Vec3IntLike, int]] = None,
        chunks_per_shard: Optional[Union[Vec3IntLike, int]] = None,
        compress: Optional[bool] = None,
        executor: Optional[Executor] = None,
    ) -> MagView:
        """
        Copies the data at `foreign_mag_view_or_path` which can belong to another dataset
        to the current dataset. Additionally, the relevant information from the
        `datasource-properties.json` of the other dataset are copied, too.
        """
        self.dataset._ensure_writable()
        foreign_mag_view = MagView._ensure_mag_view(foreign_mag_view_or_path)
        self._assert_mag_does_not_exist_yet(foreign_mag_view.mag)

        mag_view = self.add_mag(
            mag=foreign_mag_view.mag,
            chunk_shape=chunk_shape or foreign_mag_view._array_info.chunk_shape,
            chunks_per_shard=chunks_per_shard
            or foreign_mag_view._array_info.chunks_per_shard,
            compress=(
                compress
                if compress is not None
                else foreign_mag_view._array_info.compression_mode
            ),
        )

        if extend_layer_bounding_box:
            self.bounding_box = self.bounding_box.extended_by(
                foreign_mag_view.layer.bounding_box
            )

        foreign_mag_view.for_zipped_chunks(
            func_per_chunk=_copy_job,
            target_view=mag_view,
            executor=executor,
            progress_desc=f"Copying mag {mag_view.mag.to_layer_name()} from {foreign_mag_view.layer} to {mag_view.layer}",
        )

        return mag_view

    def add_symlink_mag(
        self,
        foreign_mag_view_or_path: Union[PathLike, str, MagView],
        make_relative: bool = False,
        extend_layer_bounding_box: bool = True,
    ) -> MagView:
        """
        Creates a symlink to the data at `foreign_mag_view_or_path` which belongs to another dataset.
        The relevant information from the `datasource-properties.json` of the other dataset is copied to this dataset.
        Note: If the other dataset modifies its bounding box afterwards, the change does not affect this properties
        (or vice versa).
        If make_relative is True, the symlink is made relative to the current dataset path.
        Symlinked mags can only be added to layers on local file systems.
        """
        self.dataset._ensure_writable()
        foreign_mag_view = MagView._ensure_mag_view(foreign_mag_view_or_path)
        self._assert_mag_does_not_exist_yet(foreign_mag_view.mag)

        assert is_fs_path(
            self.path
        ), f"Cannot create symlinks in remote layer {self.path}"
        assert is_fs_path(
            foreign_mag_view.path
        ), f"Cannot create symlink to remote mag {foreign_mag_view.path}"

        foreign_normalized_mag_path = (
            Path(relpath(foreign_mag_view.path, self.path))
            if make_relative
            else foreign_mag_view.path.resolve()
        )

        (self.path / str(foreign_mag_view.mag)).symlink_to(foreign_normalized_mag_path)

        mag = self.add_mag_for_existing_files(foreign_mag_view.mag)

        if extend_layer_bounding_box:
            self.bounding_box = self.bounding_box.extended_by(
                foreign_mag_view.layer.bounding_box
            )
        return mag

    def add_fs_copy_mag(
        self,
        foreign_mag_view_or_path: Union[PathLike, str, MagView],
        extend_layer_bounding_box: bool = True,
    ) -> MagView:
        """
        Copies the data at `foreign_mag_view_or_path` which belongs to another dataset to the current dataset via the filesystem.
        Additionally, the relevant information from the `datasource-properties.json` of the other dataset are copied, too.
        """
        self.dataset._ensure_writable()
        foreign_mag_view = MagView._ensure_mag_view(foreign_mag_view_or_path)
        self._assert_mag_does_not_exist_yet(foreign_mag_view.mag)

        copytree(
            foreign_mag_view.path,
            self.path / str(foreign_mag_view.mag),
        )

        mag = self.add_mag_for_existing_files(foreign_mag_view.mag)

        if extend_layer_bounding_box:
            self.bounding_box = self.bounding_box.extended_by(
                foreign_mag_view.layer.bounding_box
            )

        return mag

    def _create_dir_for_mag(
        self, mag: Union[int, str, list, tuple, np.ndarray, Mag]
    ) -> None:
        mag = Mag(mag).to_layer_name()
        full_path = self.path / mag
        full_path.mkdir(parents=True, exist_ok=True)

    def _assert_mag_does_not_exist_yet(
        self, mag: Union[int, str, list, tuple, np.ndarray, Mag]
    ) -> None:
        if mag in self.mags.keys():
            raise IndexError(
                "Adding mag {} failed. There is already a mag with this name".format(
                    mag
                )
            )

    def _get_dataset_from_align_with_other_layers(
        self, align_with_other_layers: Union[bool, "Dataset"]
    ) -> Optional["Dataset"]:
        if isinstance(align_with_other_layers, bool):
            return self.dataset if align_with_other_layers else None
        else:
            return align_with_other_layers

    def downsample(
        self,
        from_mag: Optional[Mag] = None,
        coarsest_mag: Optional[Mag] = None,
        interpolation_mode: str = "default",
        compress: bool = True,
        sampling_mode: Union[str, SamplingModes] = SamplingModes.ANISOTROPIC,
        align_with_other_layers: Union[bool, "Dataset"] = True,
        buffer_shape: Optional[Vec3Int] = None,
        force_sampling_scheme: bool = False,
        args: Optional[Namespace] = None,  # deprecated
        allow_overwrite: bool = False,
        only_setup_mags: bool = False,
        executor: Optional[Executor] = None,
    ) -> None:
        """
        Downsamples the data starting from `from_mag` until a magnification is `>= max(coarsest_mag)`.
        There are three different `sampling_modes`:

        - 'anisotropic' - The next magnification is chosen so that the width, height and depth of a downsampled voxel assimilate. For example, if the z resolution is worse than the x/y resolution, z won't be downsampled in the first downsampling step(s). As a basis for this method, the voxel_size from the datasource-properties.json is used.
        - 'isotropic' - Each dimension is downsampled equally.
        - 'constant_z' - The x and y dimensions are downsampled equally, but the z dimension remains the same.

        See `downsample_mag` for more information.

        Example:
        ```python
        from webknossos import SamplingModes

        # ...
        # let 'layer' be a `Layer` with only `Mag(1)`
        assert "1" in self.mags.keys()

        layer.downsample(
            coarsest_mag=Mag(4),
            sampling_mode=SamplingModes.ISOTROPIC
        )

        assert "2" in self.mags.keys()
        assert "4" in self.mags.keys()
        ```
        """
        if from_mag is None:
            assert (
                len(self.mags.keys()) > 0
            ), "Failed to downsample data because no existing mag was found."
            from_mag = max(self.mags.keys())

        assert (
            from_mag in self.mags.keys()
        ), f"Failed to downsample data. The from_mag ({from_mag.to_layer_name()}) does not exist."

        if coarsest_mag is None:
            coarsest_mag = calculate_default_coarsest_mag(self.bounding_box.size_xyz)

        sampling_mode = SamplingModes.parse(sampling_mode)

        if self._properties.bounding_box.size.z == 1:
            if sampling_mode != SamplingModes.CONSTANT_Z:
                warnings.warn(
                    "[INFO] The sampling_mode was changed to 'CONSTANT_Z'. Downsampling 2D data with a different sampling mode mixes in black and thus leads to darkened images."
                )
                sampling_mode = SamplingModes.CONSTANT_Z

        voxel_size: Optional[Tuple[float, float, float]] = None
        if sampling_mode == SamplingModes.ANISOTROPIC:
            voxel_size = self.dataset.voxel_size
        elif sampling_mode == SamplingModes.ISOTROPIC:
            voxel_size = None
        elif sampling_mode == SamplingModes.CONSTANT_Z:
            coarsest_mag_with_fixed_z = coarsest_mag.to_list()
            coarsest_mag_with_fixed_z[2] = from_mag.to_list()[2]
            coarsest_mag = Mag(coarsest_mag_with_fixed_z)
            voxel_size = None
        else:
            raise AttributeError(
                f"Downsampling failed: {sampling_mode} is not a valid SamplingMode ({SamplingModes.ANISOTROPIC}, {SamplingModes.ISOTROPIC}, {SamplingModes.CONSTANT_Z})"
            )

        dataset_to_align_with = self._get_dataset_from_align_with_other_layers(
            align_with_other_layers
        )
        mags_to_downsample = calculate_mags_to_downsample(
            from_mag, coarsest_mag, dataset_to_align_with, voxel_size
        )

        if len(set([max(m.to_list()) for m in mags_to_downsample])) != len(
            mags_to_downsample
        ):
            msg = (
                f"[INFO] The downsampling scheme contains multiple magnifications with the same maximum value. This is not supported by WEBKNOSSOS. "
                f"Consider using a different sampling mode (e.g. {SamplingModes.ISOTROPIC}). "
                f"The calculated downsampling scheme is: {[m.to_layer_name() for m in mags_to_downsample]}"
            )
            if force_sampling_scheme:
                warnings.warn(msg)
            else:
                raise RuntimeError(msg)

        for prev_mag, target_mag in zip(
            [from_mag] + mags_to_downsample[:-1], mags_to_downsample
        ):
            self.downsample_mag(
                from_mag=prev_mag,
                target_mag=target_mag,
                interpolation_mode=interpolation_mode,
                compress=compress,
                buffer_shape=buffer_shape,
                args=args,
                allow_overwrite=allow_overwrite,
                only_setup_mag=only_setup_mags,
                executor=executor,
            )

    def downsample_mag(
        self,
        from_mag: Mag,
        target_mag: Mag,
        interpolation_mode: str = "default",
        compress: bool = True,
        buffer_shape: Optional[Vec3Int] = None,
        args: Optional[Namespace] = None,  # deprecated
        allow_overwrite: bool = False,
        only_setup_mag: bool = False,
        executor: Optional[Executor] = None,
    ) -> None:
        """
        Performs a single downsampling step from `from_mag` to `target_mag`.

        The supported `interpolation_modes` are:

         - "median"
         - "mode"
         - "nearest"
         - "bilinear"
         - "bicubic"

        If allow_overwrite is True, an existing Mag may be overwritten.

        If only_setup_mag is True, the magnification is created, but left
        empty. This parameter can be used to prepare for parallel downsampling
        of multiple layers while avoiding parallel writes with outdated updates
        to the datasource-properties.json file.

        `executor` can be passed to allow distributed computation, parallelizing
        across chunks. `args` is deprecated.
        """
        self._dataset._ensure_writable()

        if args is not None:
            warn_deprecated(
                "args argument",
                "executor (e.g. via webknossos.utils.get_executor_for_args(args))",
            )

        assert (
            from_mag in self.mags.keys()
        ), f"Failed to downsample data. The from_mag ({from_mag.to_layer_name()}) does not exist."

        parsed_interpolation_mode = parse_interpolation_mode(
            interpolation_mode, self.category
        )

        assert from_mag <= target_mag
        assert (
            allow_overwrite or target_mag not in self.mags
        ), "The target mag already exists. Pass allow_overwrite=True if you want to overwrite it."

        prev_mag_view = self.mags[from_mag]

        mag_factors = target_mag.to_vec3_int() // from_mag.to_vec3_int()

        if target_mag in self.mags.keys() and allow_overwrite:
            target_mag_view = self.get_mag(target_mag)
        else:
            # initialize the new mag
            target_mag_view = self._initialize_mag_from_other_mag(
                target_mag, prev_mag_view, compress
            )

        if only_setup_mag:
            return

        bb_mag1 = self.bounding_box.align_with_mag(target_mag, ceil=True)

        # Get target view
        target_view = target_mag_view.get_view(absolute_bounding_box=bb_mag1)

        source_view = prev_mag_view.get_view(
            absolute_bounding_box=bb_mag1,
            read_only=True,
        )

        # perform downsampling
        with get_executor_for_args(args, executor) as executor:
            if buffer_shape is None:
                buffer_shape = determine_buffer_shape(prev_mag_view.info)
            func = named_partial(
                downsample_cube_job,
                mag_factors=mag_factors,
                interpolation_mode=parsed_interpolation_mode,
                buffer_shape=buffer_shape,
            )

            source_view.for_zipped_chunks(
                # this view is restricted to the bounding box specified in the properties
                func,
                target_view=target_view,
                executor=executor,
                progress_desc=f"Downsampling layer {self.name} from Mag {from_mag} to Mag {target_mag}",
            )

    def redownsample(
        self,
        interpolation_mode: str = "default",
        compress: bool = True,
        buffer_shape: Optional[Vec3Int] = None,
        args: Optional[Namespace] = None,  # deprecated
        executor: Optional[Executor] = None,
    ) -> None:
        """
        Use this method to recompute downsampled magnifications after mutating data in the
        base magnification.
        """

        mags = sorted(self.mags.keys(), key=lambda m: m.to_list())
        if len(mags) <= 1:
            # No downsampled magnifications exist. Thus, there's nothing to do.
            return
        from_mag = mags[0]
        target_mags = mags[1:]
        self.downsample_mag_list(
            from_mag=from_mag,
            target_mags=target_mags,
            interpolation_mode=interpolation_mode,
            compress=compress,
            buffer_shape=buffer_shape,
            args=args,
            allow_overwrite=True,
            executor=executor,
        )

    def downsample_mag_list(
        self,
        from_mag: Mag,
        target_mags: List[Mag],
        interpolation_mode: str = "default",
        compress: bool = True,
        buffer_shape: Optional[Vec3Int] = None,
        args: Optional[Namespace] = None,  # deprecated
        allow_overwrite: bool = False,
        only_setup_mags: bool = False,
        executor: Optional[Executor] = None,
    ) -> None:
        """
        Downsamples the data starting at `from_mag` to each magnification in `target_mags` iteratively.

        See `downsample_mag` for more information.
        """
        assert (
            from_mag in self.mags.keys()
        ), f"Failed to downsample data. The from_mag ({from_mag}) does not exist."

        # The lambda function is important because 'sorted(target_mags)' would only sort by the maximum element per mag
        target_mags = sorted(target_mags, key=lambda m: m.to_list())

        for i in range(len(target_mags) - 1):
            assert np.less_equal(
                target_mags[i].to_np(), target_mags[i + 1].to_np()
            ).all(), (
                f"Downsampling failed: cannot downsample {target_mags[i].to_layer_name()} to {target_mags[i + 1].to_layer_name()}. "
                f"Check 'target_mags' ({', '.join([str(mag) for mag in target_mags])}): each pair of adjacent Mags results in a downsampling step."
            )

        source_mag = from_mag
        for target_mag in target_mags:
            self.downsample_mag(
                source_mag,
                target_mag,
                interpolation_mode=interpolation_mode,
                compress=compress,
                buffer_shape=buffer_shape,
                args=args,
                allow_overwrite=allow_overwrite,
                only_setup_mag=only_setup_mags,
                executor=executor,
            )
            source_mag = target_mag

    def upsample(
        self,
        from_mag: Mag,
        finest_mag: Mag = Mag(1),
        compress: bool = False,
        sampling_mode: Union[str, SamplingModes] = SamplingModes.ANISOTROPIC,
        align_with_other_layers: Union[bool, "Dataset"] = True,
        buffer_shape: Optional[Vec3Int] = None,
        buffer_edge_len: Optional[int] = None,
        args: Optional[Namespace] = None,  # deprecated
        executor: Optional[Executor] = None,
        *,
        min_mag: Optional[Mag] = None,
    ) -> None:
        """
        Upsamples the data starting from `from_mag` as long as the magnification is `>= finest_mag`.
        There are three different `sampling_modes`:

        - 'anisotropic' - The next magnification is chosen so that the width, height and depth of a downsampled voxel assimilate. For example, if the z resolution is worse than the x/y resolution, z won't be downsampled in the first downsampling step(s). As a basis for this method, the voxel_size from the datasource-properties.json is used.
        - 'isotropic' - Each dimension is downsampled equally.
        - 'constant_z' - The x and y dimensions are downsampled equally, but the z dimension remains the same.

        `min_mag` is deprecated, please use `finest_mag` instead.
        """
        self._dataset._ensure_writable()

        if args is not None:
            warn_deprecated(
                "args argument",
                "executor (e.g. via webknossos.utils.get_executor_for_args(args))",
            )

        assert (
            from_mag in self.mags.keys()
        ), f"Failed to upsample data. The from_mag ({from_mag.to_layer_name()}) does not exist."

        if min_mag is not None:
            warn_deprecated("upsample(min_mag=…)", "upsample(finest_mag=…)")
            assert finest_mag == Mag(
                1
            ), "Cannot set both min_mag and finest_mag, please only use finest_mag."
            finest_mag = min_mag

        sampling_mode = SamplingModes.parse(sampling_mode)

        voxel_size: Optional[Tuple[float, float, float]] = None
        if sampling_mode == SamplingModes.ANISOTROPIC:
            voxel_size = self.dataset.voxel_size
        elif sampling_mode == SamplingModes.ISOTROPIC:
            voxel_size = None
        elif sampling_mode == SamplingModes.CONSTANT_Z:
            finest_mag_with_fixed_z = finest_mag.to_list()
            finest_mag_with_fixed_z[2] = from_mag.to_list()[2]
            finest_mag = Mag(finest_mag_with_fixed_z)
            voxel_size = None
        else:
            raise AttributeError(
                f"Upsampling failed: {sampling_mode} is not a valid UpsamplingMode ({SamplingModes.ANISOTROPIC}, {SamplingModes.ISOTROPIC}, {SamplingModes.CONSTANT_Z})"
            )

        if buffer_shape is None and buffer_edge_len is not None:
            buffer_shape = Vec3Int.full(buffer_edge_len)

        dataset_to_align_with = self._get_dataset_from_align_with_other_layers(
            align_with_other_layers
        )
        mags_to_upsample = calculate_mags_to_upsample(
            from_mag, finest_mag, dataset_to_align_with, voxel_size
        )

        for prev_mag, target_mag in zip(
            [from_mag] + mags_to_upsample[:-1], mags_to_upsample
        ):
            assert prev_mag > target_mag
            assert target_mag not in self.mags

            prev_mag_view = self.mags[prev_mag]

            mag_factors = [
                t / s for (t, s) in zip(target_mag.to_list(), prev_mag.to_list())
            ]

            # initialize the new mag
            target_mag_view = self._initialize_mag_from_other_mag(
                target_mag, prev_mag_view, compress
            )

            # We need to make sure the layer's bounding box is aligned
            # with the previous mag. Otherwise, `for_zipped_chunks` will fail.
            # Saving the original layer bbox for later restore
            old_layer_bbox = self.bounding_box
            self.bounding_box = prev_mag_view.bounding_box
            bbox_mag1 = self.bounding_box.align_with_mag(prev_mag, ceil=True)
            # Get target view
            target_view = target_mag_view.get_view(absolute_bounding_box=bbox_mag1)

            # perform upsampling
            with get_executor_for_args(args, executor) as actual_executor:
                if buffer_shape is None:
                    buffer_shape = determine_buffer_shape(prev_mag_view.info)
                func = named_partial(
                    upsample_cube_job,
                    mag_factors=mag_factors,
                    buffer_shape=buffer_shape,
                )
                prev_mag_view.get_view(
                    absolute_bounding_box=bbox_mag1
                ).for_zipped_chunks(
                    # this view is restricted to the bounding box specified in the properties
                    func,
                    target_view=target_view,
                    executor=actual_executor,
                    progress_desc=f"Upsampling from Mag {prev_mag} to Mag {target_mag}",
                )
            # Restoring the original layer bbox
            self.bounding_box = old_layer_bbox

    def _setup_mag(self, mag: Mag, path: Optional[str] = None) -> None:
        # This method is used to initialize the mag when opening the Dataset. This does not create e.g. the wk_header.

        mag_name = mag.to_layer_name()

        self._assert_mag_does_not_exist_yet(mag)

        try:
            cls_array = BaseArray.get_class(self._properties.data_format)
            info = cls_array.open(
                _find_mag_path_on_disk(self.dataset.path, self.name, mag_name, path)
            ).info
            self._mags[mag] = MagView(
                self,
                mag,
                info.chunk_shape,
                info.chunks_per_shard,
                info.compression_mode,
            )
            self._mags[mag]._read_only = self._dataset.read_only
        except ArrayException:
            logging.exception(
                f"Failed to setup magnification {mag_name}, which is specified in the datasource-properties.json:"
            )

    def _initialize_mag_from_other_mag(
        self, new_mag_name: Union[str, Mag], other_mag: MagView, compress: bool
    ) -> MagView:
        return self.add_mag(
            new_mag_name,
            chunk_shape=other_mag.info.chunk_shape,
            chunks_per_shard=other_mag.info.chunks_per_shard,
            compress=compress,
        )

    def __repr__(self) -> str:
        return f"Layer({repr(self.name)}, dtype_per_channel={self.dtype_per_channel}, num_channels={self.num_channels})"

    def _get_largest_segment_id_maybe(self) -> Optional[int]:
        return None

    @classmethod
    def _ensure_layer(cls, layer: Union[str, PathLike, "Layer"]) -> "Layer":
        if isinstance(layer, Layer):
            return layer
        else:
            # local import to prevent circular dependency
            from .dataset import Dataset

            layer_path = UPath(layer)
            return Dataset.open(layer_path.parent).get_layer(layer_path.name)


class SegmentationLayer(Layer):
    _properties: SegmentationLayerProperties

    @property
    def largest_segment_id(self) -> Optional[int]:
        return self._properties.largest_segment_id

    @largest_segment_id.setter
    def largest_segment_id(self, largest_segment_id: Optional[int]) -> None:
        self.dataset._ensure_writable()
        if largest_segment_id is not None and not isinstance(largest_segment_id, int):
            assert (
                largest_segment_id == int(largest_segment_id)
            ), f"A non-integer value was passed for largest_segment_id ({largest_segment_id})."
            largest_segment_id = int(largest_segment_id)

        self._properties.largest_segment_id = largest_segment_id
        self.dataset._export_as_json()

    @property
    def category(self) -> LayerCategoryType:
        return SEGMENTATION_CATEGORY

    def _get_largest_segment_id_maybe(self) -> Optional[int]:
        return self.largest_segment_id

    def _get_largest_segment_id(self, view: View) -> int:
        return np.max(view.read(), initial=0)

    def refresh_largest_segment_id(
        self, chunk_shape: Optional[Vec3Int] = None, executor: Optional[Executor] = None
    ) -> None:
        """Sets the largest segment id to the highest value in the data.
        largest_segment_id is set to `None` if the data is empty."""

        try:
            chunk_results = self.get_finest_mag().map_chunk(
                self._get_largest_segment_id,
                chunk_shape=chunk_shape,
                executor=executor,
                progress_desc="Searching largest segment id",
            )
            self.largest_segment_id = max(chunk_results)
        except ValueError:
            self.largest_segment_id = None
