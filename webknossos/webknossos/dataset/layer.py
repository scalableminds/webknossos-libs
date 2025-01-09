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

from webknossos.client.context import _get_context
from webknossos.dataset.defaults import SSL_CONTEXT

from ..geometry import Mag, NDBoundingBox, Vec3Int, Vec3IntLike
from ._array import ArrayException, TensorStoreArray
from ._downsampling_utils import (
    calculate_default_coarsest_mag,
    calculate_mags_to_downsample,
    calculate_mags_to_upsample,
    determine_buffer_shape,
    downsample_cube_job,
    parse_interpolation_mode,
)
from ._upsampling_utils import upsample_cube_job
from .data_format import DataFormat
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
    LazyPath,
    copytree,
    get_executor_for_args,
    is_fs_path,
    is_remote_path,
    movetree,
    named_partial,
    rmtree,
    strip_trailing_slash,
    warn_deprecated,
)
from .defaults import (
    DEFAULT_CHUNK_SHAPE,
    DEFAULT_CHUNKS_PER_SHARD,
    DEFAULT_CHUNKS_PER_SHARD_ZARR,
)
from .mag_view import MagView, _find_mag_path


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
    def __init__(self, dataset: "Dataset", properties: LayerProperties) -> None:
        """A Layer represents a portion of hierarchical data at multiple magnifications.

        A Layer consists of multiple MagViews, which store the same data in different magnifications.
        Layers are components of a Dataset and provide access to the underlying data arrays.

        Attributes:
            name (str): Name identifier for this layer
            dataset (Dataset): Parent dataset containing this layer
            path (Path): Filesystem path to this layer's data
            category (LayerCategoryType): Category of data (e.g. color, segmentation)
            dtype_per_layer (str): Data type used for the entire layer
            dtype_per_channel (np.dtype): Data type used per channel
            num_channels (int): Number of channels in the layer
            data_format (DataFormat): Format used to store the data
            default_view_configuration (Optional[LayerViewConfiguration]): View configuration
            read_only (bool): Whether layer is read-only
            mags (Dict[Mag, MagView]): Dictionary of magnification levels

        Args:
            dataset (Dataset): The parent dataset that contains this layer
            properties (LayerProperties): Properties defining this layer's attributes. Must contain num_channels.

        Raises:
            AssertionError: If properties.num_channels is None

        Note:
            Do not use this constructor manually. Instead use Dataset.add_layer() to create a Layer.
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

        for mag in properties.mags:
            self._setup_mag(Mag(mag.mag), mag.path)
        self._properties.mags = [
            res for res in self._properties.mags if Mag(res.mag) in self._mags
        ]

    @property
    def path(self) -> Path:
        """Gets the filesystem path to this layer's data.

        The path is determined from the first mag's path parent directory if mags have paths,
        otherwise uses the dataset path combined with layer name. Remote paths are handled
        specially.

        Returns:
            Path: Filesystem path to this layer's data directory

        Raises:
            AssertionError: If mags in layer point to different layers
        """

        # Assume that all mags belong to the same layer. If they have a path use them as this layers path.
        # This is necessary for remote layer / mag support.
        maybe_mag_path_str = (
            self._properties.mags[0].path if len(self._properties.mags) > 0 else None
        )
        maybe_layer_path = (
            strip_trailing_slash(UPath(maybe_mag_path_str)).parent
            if maybe_mag_path_str
            else None
        )
        for mag in self._properties.mags:
            is_same_layer = (
                mag.path is None and maybe_layer_path is None
            ) or strip_trailing_slash(UPath(mag.path)).parent == maybe_layer_path
            assert is_same_layer, "All mags of a layer must point to the same layer."
        is_remote = maybe_layer_path and is_remote_path(maybe_layer_path)
        return (
            maybe_layer_path
            if maybe_layer_path and is_remote
            else self.dataset.path / self.name
        )

    @property
    def is_remote_to_dataset(self) -> bool:
        """Whether this layer's data is stored remotely relative to its dataset.

        Returns:
            bool: True if layer path parent differs from dataset path
        """
        return self.path.parent != self.dataset.path

    @property
    def _properties(self) -> LayerProperties:
        """Gets the LayerProperties object containing layer attributes.

        Returns:
            LayerProperties: Properties object for this layer

        Note:
            Internal property used to access underlying properties storage.
        """

        return next(
            layer_property
            for layer_property in self.dataset._properties.data_layers
            if layer_property.name == self.name
        )

    @property
    def name(self) -> str:
        """Gets the name identifier of this layer.

        Returns:
            str: Layer name
        """

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
            if not mag.is_remote_to_dataset:
                mag_path_maybe = (
                    LazyPath.resolved(UPath(mag._properties.path))
                    if mag._properties.path is not None
                    else None
                )
                mag._path = _find_mag_path(
                    self.dataset.path, self.name, mag.name, mag_path_maybe
                )
            # Deleting the dataset will close the file handle.
            # The new dataset will be opened automatically when needed.
            del mag._array

        self.dataset._export_as_json()

    @property
    def dataset(self) -> "Dataset":
        """Gets the dataset containing this layer.

        Returns:
            Dataset: Parent dataset object
        """

        return self._dataset

    @property
    def bounding_box(self) -> NDBoundingBox:
        """Gets the bounding box encompassing this layer's data.

        Returns:
            NDBoundingBox: Bounding box with layer dimensions
        """

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
        """Gets the category type of this layer.

        Returns:
            LayerCategoryType: Layer category (e.g. COLOR_CATEGORY)
        """

        return COLOR_CATEGORY

    @property
    def dtype_per_layer(self) -> str:
        """Gets the data type used for the entire layer.

        Returns:
            str: Data type string (e.g. "uint8")
        """

        return _dtype_per_channel_to_dtype_per_layer(
            self.dtype_per_channel, self.num_channels
        )

    @property
    def dtype_per_channel(self) -> np.dtype:
        """Gets the data type used per channel.

        Returns:
            np.dtype: NumPy data type for individual channels
        """

        return self._dtype_per_channel

    @property
    def num_channels(self) -> int:
        """Gets the number of channels in this layer.

        Returns:
            int: Number of channels

        Raises:
            AssertionError: If num_channels is not set in properties
        """

        assert self._properties.num_channels is not None
        return self._properties.num_channels

    @property
    def data_format(self) -> DataFormat:
        """Gets the data storage format used by this layer.

        Returns:
            DataFormat: Format used to store data

        Raises:
            AssertionError: If data_format is not set in properties
        """

        assert self._properties.data_format is not None
        return self._properties.data_format

    @property
    def default_view_configuration(self) -> Optional[LayerViewConfiguration]:
        """Gets the default view configuration for this layer.

        Returns:
            Optional[LayerViewConfiguration]: View configuration if set, otherwise None
        """

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
        """Whether this layer is read-only.

        Returns:
            bool: True if layer is read-only, False if writable
        """
        return self.dataset.read_only

    @property
    def mags(self) -> Dict[Mag, MagView]:
        """
        Getter for dictionary containing all mags.
        """
        return self._mags

    def get_mag(self, mag: Union[int, str, list, tuple, np.ndarray, Mag]) -> MagView:
        """Gets the MagView for the specified magnification level.

        Returns a view of the data at the requested magnification level. The mag
        parameter can be specified in various formats that will be normalized.

        Args:
            mag: Magnification identifier in multiple formats (int, str, list, etc)

        Returns:
            MagView: View of data at the specified magnification

        Raises:
            IndexError: If specified magnification does not exist
        """
        mag = Mag(mag)
        if mag not in self.mags.keys():
            raise IndexError(
                "The mag {} is not a mag of this layer".format(mag.to_layer_name())
            )
        return self.mags[mag]

    def get_finest_mag(self) -> MagView:
        """Gets the MagView with the finest/smallest magnification.

        Returns:
            MagView: View of data at finest available magnification
        """
        return self.get_mag(min(self.mags.keys()))

    def get_best_mag(self) -> MagView:
        """Gets the finest magnification view (deprecated).

        Deprecated method name. Please use get_finest_mag() instead.

        Returns:
            MagView: View of finest magnification level.

        Deprecated:
            Use get_finest_mag() instead.
        """
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
        """Creates and adds a new magnification level to the layer.

        The new magnification can be configured with various storage parameters to
        optimize performance, notably `chunk_shape`, `chunks_per_shard` and `compress`. Note that writing compressed data which is not aligned with the blocks on disk may result in
        diminished performance, as full blocks will automatically be read to pad the write actions. Alternatively,
        you can call mag.compress() after all the data was written

        Args:
            mag: Identifier for new magnification level
            chunk_shape: Shape of chunks for storage. Recommended (32,32,32) or (64,64,64)
            chunks_per_shard: Number of chunks per shard file
            compress: Whether to enable compression
            chunk_size: Deprecated, use chunk_shape
            block_len: Deprecated, use chunk_shape
            file_len: Deprecated, use chunks_per_shard

        Returns:
            MagView: View of newly created magnification level

        Raises:
            IndexError: If magnification already exists
            Warning: If chunk_shape is not optimal for WEBKNOSSOS performance
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
        mag_path = LazyPath.resolved(self._create_dir_for_mag(mag))

        mag_view = MagView.create(
            self,
            mag,
            chunk_shape=chunk_shape,
            chunks_per_shard=chunks_per_shard,
            compression_mode=compression_mode,
            path=mag_path,
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
        """Creates a MagView for existing data files.

        Adds a magnification level by linking to data files that already exist
        on the filesystem.

        Args:
            mag: Identifier for magnification level

        Returns:
            MagView: View of existing magnification data

        Raises:
            AssertionError: If magnification already exists in layer
            ArrayException: If files cannot be opened as valid arrays
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

    def add_existing_remote_mag_view(
        self,
        mag_view_maybe: Union[int, str, list, tuple, np.ndarray, Mag, MagView],
    ) -> MagView:
        """Adds a remote magnification view to this layer.

        Links a magnification from a remote dataset into this layer. The remote
        mag must already exist.

        Args:
            mag_view_maybe: Remote magnification to add, as view or identifier

        Returns:
            MagView: View of the added remote magnification

        Raises:
            AssertionError: If magnification exists or remote mag invalid
            ArrayException: If remote data cannot be accessed
        """

        self.dataset._ensure_writable()
        mag_path = (
            mag_view_maybe.path
            if isinstance(mag_view_maybe, MagView)
            else self.path / Mag(mag_view_maybe).to_layer_name()
        )
        mag = (
            mag_view_maybe.mag
            if isinstance(mag_view_maybe, MagView)
            else Mag(mag_view_maybe)
        )
        mag_view = (
            mag_view_maybe
            if isinstance(mag_view_maybe, MagView)
            else MagView._ensure_mag_view(mag_path)
        )
        assert (
            mag not in self.mags
        ), f"Cannot add mag {mag} as it already exists for layer {self}"
        self._setup_mag(mag, mag_path)
        self._properties.mags.append(mag_view._properties)
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
            if (
                chunk_shape is not None
                and self._mags[mag].info.chunk_shape != chunk_shape
            ):
                raise ValueError(
                    f"Cannot get_or_add_mag: The mag {mag} already exists, but the chunk sizes do not match"
                )
            if (
                chunks_per_shard is not None
                and self._mags[mag].info.chunks_per_shard != chunks_per_shard
            ):
                raise ValueError(
                    f"Cannot get_or_add_mag: The mag {mag} already exists, but the chunks per shard do not match"
                )
            if (
                compression_mode is not None
                and self._mags[mag].info.compression_mode != compression_mode
            ):
                raise ValueError(
                    f"Cannot get_or_add_mag: The mag {mag} already exists, but the compression modes do not match"
                )
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

        full_path = self._mags[mag].path
        del self._mags[mag]
        self._properties.mags = [
            res for res in self._properties.mags if Mag(res.mag) != mag
        ]
        self.dataset._export_as_json()
        # delete files on disk
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
            chunk_shape=chunk_shape or foreign_mag_view.info.chunk_shape,
            chunks_per_shard=chunks_per_shard or foreign_mag_view.info.chunks_per_shard,
            compress=(
                compress
                if compress is not None
                else foreign_mag_view.info.compression_mode
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

    def add_remote_mag(
        self,
        foreign_mag_view_or_path: Union[PathLike, str, MagView],
        extend_layer_bounding_box: bool = True,
    ) -> MagView:
        """
        Adds the mag at `foreign_mag_view_or_path` which belongs to foreign dataset.
        The relevant information from the `datasource-properties.json` of the other dataset is copied to this dataset.
        Note: If the other dataset modifies its bounding box afterwards, the change does not affect this properties
        (or vice versa).
        """
        self.dataset._ensure_writable()
        foreign_mag_view = MagView._ensure_mag_view(foreign_mag_view_or_path)
        self._assert_mag_does_not_exist_yet(foreign_mag_view.mag)

        assert is_remote_path(
            foreign_mag_view.path
        ), f"Cannot create foreign mag for local mag {foreign_mag_view.path}. Please use layer.add_mag in this case."
        assert self.data_format == foreign_mag_view.info.data_format, (
            f"Cannot add a remote mag whose data format {foreign_mag_view.info.data_format} "
            + f"does not match the layers data format {self.data_format}"
        )
        assert self.dtype_per_channel == foreign_mag_view.get_dtype(), (
            f"The dtype/elementClass of the remote mag {foreign_mag_view.get_dtype()} "
            + f"must match the layer's dtype {self.dtype_per_channel}"
        )

        mag = self.add_existing_remote_mag_view(foreign_mag_view)

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

    def add_mag_from_zarrarray(
        self,
        mag: Union[int, str, list, tuple, np.ndarray, Mag],
        path: PathLike,
        move: bool = False,
        extend_layer_bounding_box: bool = True,
    ) -> MagView:
        """
        Copies the data at `path` to the current layer of the dataset
        via the filesystem and adds it as `mag`. When `move` flag is set
        the array is moved, otherwise a copy of the zarrarray is created.
        """
        self.dataset._ensure_writable()
        source_path = Path(path)

        try:
            TensorStoreArray.open(source_path)
        except ArrayException as e:
            raise ValueError(
                "The given path does not lead to a valid Zarr Array: "
            ) from e
        else:
            mag = Mag(mag)
            self._assert_mag_does_not_exist_yet(mag)
            if move:
                movetree(source_path, self.path / str(mag))
            else:
                copytree(source_path, self.path / str(mag))

            mag_view = self.add_mag_for_existing_files(mag)

            if extend_layer_bounding_box:
                # assumption: the topleft of the bounding box is still the same, the size might differ
                # axes of the layer and the zarr array provided are the same
                zarray_size = (
                    mag_view.info.shape[mag_view.info.dimension_names.index(axis)]
                    for axis in self.bounding_box.axes
                    if axis != "c"
                )
                size = self.bounding_box.size.pairmax(zarray_size)
                self.bounding_box = self.bounding_box.with_size(size)

            return mag_view

    def _create_dir_for_mag(
        self, mag: Union[int, str, list, tuple, np.ndarray, Mag]
    ) -> Path:
        mag_name = Mag(mag).to_layer_name()
        full_path = self.path / mag_name
        full_path.mkdir(parents=True, exist_ok=True)
        return full_path

    def _assert_mag_does_not_exist_yet(
        self, mag: Union[int, str, list, tuple, np.ndarray, Mag]
    ) -> None:
        """Verifies a magnification does not already exist.

        Args:
            mag: Magnification to check

        Raises:
            IndexError: If magnification exists
        """
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
        """Downsample data from a source magnification to coarser magnifications.

        Downsamples the data starting from from_mag until a magnification is >= max(coarsest_mag).
        Different sampling modes control how dimensions are downsampled.

        Args:
            from_mag (Optional[Mag]): Source magnification to downsample from. Defaults to highest existing mag.
            coarsest_mag (Optional[Mag]): Target magnification to stop at. Defaults to calculated value.
            interpolation_mode (str): Interpolation method to use. Defaults to "default".
                Supported modes: "median", "mode", "nearest", "bilinear", "bicubic"
            compress (bool): Whether to compress the generated magnifications. Defaults to True.
            sampling_mode (Union[str, SamplingModes]): How dimensions should be downsampled.
                Defaults to ANISOTROPIC.
            align_with_other_layers (Union[bool, Dataset]): Whether to align with other layers. True by default.
            buffer_shape (Optional[Vec3Int]): Shape of processing buffer. Defaults to None.
            force_sampling_scheme (bool): Force invalid sampling schemes. Defaults to False.
            args (Optional[Namespace]): Deprecated argument handler.
            allow_overwrite (bool): Whether existing mags can be overwritten. False by default.
            only_setup_mags (bool): Only create mags without data. False by default.
            executor (Optional[Executor]): Executor for parallel processing. None by default.

        Raises:
            AssertionError: If from_mag does not exist
            RuntimeError: If sampling scheme produces invalid magnifications
            AttributeError: If sampling_mode is invalid

        Examples:
            ```python
            from webknossos import SamplingModes

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
        """Performs a single downsampling step between magnification levels.

        Args:
            from_mag: Source magnification level
            target_mag: Target magnification level
            interpolation_mode: Method for interpolation ("median", "mode", "nearest", "bilinear", "bicubic")
            compress: Whether to compress target data
            buffer_shape: Shape of processing buffer
            args: Deprecated, use executor
            allow_overwrite: Whether to allow overwriting existing mag
            only_setup_mag: Only create mag without data. This parameter can be used to prepare for parallel downsampling of multiple layers while avoiding parallel writes with outdated updates to the datasource-properties.json file.
            executor: Executor for parallel processing

        Raises:
            AssertionError: If from_mag doesn't exist or target exists without overwrite"""

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
        """Recompute all downsampled magnifications from base mag.

        Used after modifying data in the base magnification to update
        all derived magnifications.

        Args:
            interpolation_mode: Method for interpolation
            compress: Whether to compress recomputed data
            buffer_shape: Shape of processing buffer
            args: Deprecated, use executor
            executor: Executor for parallel processing
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
        """Downsample data iteratively through multiple magnification levels.

        Performs sequential downsampling from from_mag through each magnification
        in target_mags in order.

        Args:
            from_mag (Mag): Source magnification to start from
            target_mags (List[Mag]): Ordered list of target magnifications
            interpolation_mode (str): Interpolation method to use. Defaults to "default".
            compress (bool): Whether to compress outputs. Defaults to True.
            buffer_shape (Optional[Vec3Int]): Shape of processing buffer.
            args (Optional[Namespace]): Deprecated, use executor.
            allow_overwrite (bool): Whether to allow overwriting mags. Defaults to False.
            only_setup_mags (bool): Only create mag structures without data. Defaults to False.
            executor (Optional[Executor]): Executor for parallel processing.

        Raises:
            AssertionError: If from_mag doesn't exist or target mags not in ascending order

        See downsample_mag() for more details on parameters.
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
        """Upsample data to finer magnifications.

        Upsamples from a coarser magnification to a sequence of finer magnifications,
        stopping at finest_mag. The sampling mode controls how dimensions are handled.

        Args:
            from_mag (Mag): Source coarse magnification
            finest_mag (Mag): Target finest magnification (default Mag(1))
            compress (bool): Whether to compress upsampled data. Defaults to False.
            sampling_mode (Union[str, SamplingModes]): How dimensions should be upsampled:
                - 'anisotropic': Equalizes voxel dimensions based on voxel_size
                - 'isotropic': Equal upsampling in all dimensions
                - 'constant_z': Only upsamples x/y dimensions. z remains unchanged.
            align_with_other_layers: Whether to align mags with others. Defaults to True.
            buffer_shape (Optional[Vec3Int]): Shape of processing buffer.
            buffer_edge_len (Optional[int]): Deprecated, use buffer_shape.
            args (Optional[Namespace]): Deprecated, use executor.
            executor (Optional[Executor]): Executor for parallel processing.
            min_mag (Optional[Mag]): Deprecated, use finest_mag.

        Raises:
            AssertionError: If from_mag doesn't exist or finest_mag invalid
            AttributeError: If sampling_mode is invalid
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

    def _setup_mag(self, mag: Mag, path: Optional[Union[str, PathLike]] = None) -> None:
        """Initialize a magnification level when opening the Dataset.

        Does not create storage headers/metadata, e.g. wk_header.

        Args:
            mag: Magnification level to setup
            path: Optional path override for mag data

        Raises:
            ArrayException: If mag setup fails
        """

        mag_name = mag.to_layer_name()

        self._assert_mag_does_not_exist_yet(mag)
        # To setup the mag for non-public remote paths, we need to get the token from the context
        if path is not None and is_remote_path(UPath(path)):
            wk_context = _get_context()
            token = wk_context.datastore_token
        else:
            token = None
        mag_path_maybe = (
            LazyPath.resolved(
                UPath(
                    path,
                    headers={} if token is None else {"X-Auth-Token": token},
                    ssl=SSL_CONTEXT,
                )
            )
            if path is not None
            else None
        )
        try:
            self._mags[mag] = MagView(
                self,
                mag,
                _find_mag_path(self.dataset.path, self.name, mag_name, mag_path_maybe),
            )
            self._mags[mag]._read_only = self._dataset.read_only
        except ArrayException:
            logging.exception(
                f"Failed to setup magnification {mag_name}, which is specified in the datasource-properties.json:"
            )

    def _initialize_mag_from_other_mag(
        self, new_mag_name: Union[str, Mag], other_mag: MagView, compress: bool
    ) -> MagView:
        """Creates a new magnification based on settings from existing mag.

        Args:
            new_mag_name: Name/identifier for new mag
            other_mag: Existing mag to copy settings from
            compress: Whether to enable compression

        Returns:
            MagView: View of newly created magnification
        """
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
            # if is_remote_path(layer_path):
            #     return Dataset.open_remote(str(layer_path.parent)).get_layer(
            #         layer_path.name
            #     )
            return Dataset.open(layer_path.parent).get_layer(layer_path.name)


class SegmentationLayer(Layer):
    """A specialized Layer subclass for segmentation data.

    A SegmentationLayer extends the base Layer class with functionality specific
    to segmentation data, such as tracking the largest segment ID. The key
    differences are:

    - Always uses the SEGMENTATION_CATEGORY category type
    - Tracks the largest segment ID present in the data
    - Provides methods for updating the largest segment ID

    Attributes:
        largest_segment_id (Optional[int]): Highest segment ID present in data, or None if empty
        category (LayerCategoryType): Always SEGMENTATION_CATEGORY for this class

    Note:
        When creating a new SegmentationLayer, use Dataset.add_layer() rather than
        instantiating directly.
    """

    _properties: SegmentationLayerProperties

    @property
    def largest_segment_id(self) -> Optional[int]:
        """Gets the largest segment ID present in the data.

        The largest segment ID is the highest numerical identifier assigned to any
        segment in this layer. This is useful for:
        - Allocating new segment IDs
        - Validating segment ID ranges
        - Optimizing data structures

        Returns:
            Optional[int]: The highest segment ID present, or None if no segments exist
        """
        return self._properties.largest_segment_id

    @largest_segment_id.setter
    def largest_segment_id(self, largest_segment_id: Optional[int]) -> None:
        """Sets the largest segment ID.

        Updates the stored largest segment ID value and persists it to properties.

        Args:
            largest_segment_id (Optional[int]): New largest segment ID value to set.
                Pass None to indicate no segments exist.

        Raises:
            AssertionError: If value is not None and cannot be converted to an integer.
        """

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
        """Gets the largest segment ID within a view.

        Args:
            view: View of segmentation data

        Returns:
            int: Maximum segment ID value found
        """
        return np.max(view.read(), initial=0)

    def refresh_largest_segment_id(
        self, chunk_shape: Optional[Vec3Int] = None, executor: Optional[Executor] = None
    ) -> None:
        """Updates largest_segment_id based on actual data content.

        Scans through the data to find the highest segment ID value.
        Sets to None if data is empty.

        Args:
            chunk_shape: Shape of chunks for processing
            executor: Executor for parallel processing
        """

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
