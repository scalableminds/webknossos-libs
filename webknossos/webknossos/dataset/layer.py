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

from webknossos.geometry import BoundingBox, Mag, Vec3Int, Vec3IntLike

from ._array import ArrayException, BaseArray, DataFormat
from .downsampling_utils import (
    SamplingModes,
    calculate_default_max_mag,
    calculate_mags_to_downsample,
    calculate_mags_to_upsample,
    determine_buffer_shape,
    downsample_cube_job,
    parse_interpolation_mode,
)
from .layer_categories import COLOR_CATEGORY, SEGMENTATION_CATEGORY, LayerCategoryType
from .properties import (
    LayerProperties,
    LayerViewConfiguration,
    MagViewProperties,
    SegmentationLayerProperties,
    _properties_floating_type_to_python_type,
    _python_floating_type_to_properties_type,
)
from .upsampling_utils import upsample_cube_job

if TYPE_CHECKING:
    from .dataset import Dataset

from ..utils import (
    copytree,
    get_executor_for_args,
    is_fs_path,
    make_upath,
    named_partial,
    rmtree,
    warn_deprecated,
)
from .defaults import (
    DEFAULT_CHUNK_SIZE,
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
    dtype: Union[str, np.dtype],
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


def _normalize_dtype_per_channel(
    dtype_per_channel: Union[str, np.dtype, type]
) -> np.dtype:
    try:
        return np.dtype(dtype_per_channel)
    except TypeError as e:
        raise TypeError(
            "Cannot add layer. The specified 'dtype_per_channel' must be a valid dtype."
        ) from e


def _normalize_dtype_per_layer(
    dtype_per_layer: Union[str, np.dtype, type]
) -> Union[str, np.dtype]:
    try:
        dtype_per_layer = str(np.dtype(dtype_per_layer))
    except Exception:
        pass  # casting to np.dtype fails if the user specifies a special dtype like "uint24"
    return dtype_per_layer  # type: ignore[return-value]


def _dtype_per_layer_to_dtype_per_channel(
    dtype_per_layer: Union[str, np.dtype], num_channels: int
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
    dtype_per_channel: Union[str, np.dtype], num_channels: int
) -> str:
    return _convert_dtypes(
        np.dtype(dtype_per_channel),
        num_channels,
        dtype_per_layer_to_dtype_per_channel=False,
    )


def _dtype_per_channel_to_element_class(
    dtype_per_channel: Union[str, np.dtype], num_channels: int
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
    chunk_size: Optional[Union[Vec3IntLike, int]],
    chunks_per_shard: Optional[Union[Vec3IntLike, int]],
    block_len: Optional[int],
    file_len: Optional[int],
) -> Tuple[Optional[Vec3Int], Optional[Vec3Int]]:
    if chunk_size is not None:
        chunk_size = Vec3Int.from_vec_or_int(chunk_size)
    elif block_len is not None:
        warn_deprecated("block_len", "chunk_size")
        chunk_size = Vec3Int.full(block_len)

    if chunks_per_shard is not None:
        chunks_per_shard = Vec3Int.from_vec_or_int(chunks_per_shard)
    elif file_len is not None:
        warn_deprecated("file_len", "chunks_per_shard")
        chunks_per_shard = Vec3Int.full(file_len)

    return (chunk_size, chunks_per_shard)


class Layer:
    """
    A `Layer` consists of multiple `webknossos.dataset.mag_view.MagView`s, which store the same data in different magnifications.
    """

    def __init__(self, dataset: "Dataset", properties: LayerProperties) -> None:
        """
        Do not use this constructor manually. Instead use `webknossos.dataset.layer.Dataset.add_layer` to create a `Layer`.
        """
        # It is possible that the properties on disk do not contain the number of channels.
        # Therefore, the parameter is optional. However at this point, 'num_channels' was already inferred.
        assert properties.num_channels is not None

        self._name: str = (
            properties.name
        )  # The name is also stored in the properties, but the name is required to get the properties.
        self._dataset = dataset
        self._dtype_per_channel = _element_class_to_dtype_per_channel(
            properties.element_class, properties.num_channels
        )
        self._mags: Dict[Mag, MagView] = {}

        self.path.mkdir(parents=True, exist_ok=True)

        for mag in properties.mags:
            self._setup_mag(Mag(mag.mag))
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
        assert (
            layer_name not in self.dataset.layers.keys()
        ), f"Failed to rename layer {self.name} to {layer_name}: The new name already exists."
        assert is_fs_path(self.path), f"Cannot rename remote layer {self.path}"
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
        self._properties.default_view_configuration = view_configuration
        self.dataset._export_as_json()  # update properties on disk

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

    def get_best_mag(self) -> MagView:

        return self.get_mag(min(self.mags.keys()))

    def add_mag(
        self,
        mag: Union[int, str, list, tuple, np.ndarray, Mag],
        chunk_size: Optional[Union[Vec3IntLike, int]] = None,  # DEFAULT_CHUNK_SIZE,
        chunks_per_shard: Optional[
            Union[int, Vec3IntLike]
        ] = None,  # DEFAULT_CHUNKS_PER_SHARD,
        compress: bool = False,
        block_len: Optional[int] = None,  # deprecated
        file_len: Optional[int] = None,  # deprecated
    ) -> MagView:
        """
        Creates a new mag called and adds it to the layer.
        The parameter `chunk_size`, `chunks_per_shard` and `compress` can be
        specified to adjust how the data is stored on disk.
        Note that writing compressed data which is not aligned with the blocks on disk may result in
        diminished performance, as full blocks will automatically be read to pad the write actions. Alternatively,
        you can call mag.compress() after all the data was written

        The return type is `webknossos.dataset.mag_view.MagView`.

        Raises an IndexError if the specified `mag` already exists.
        """
        # normalize the name of the mag
        mag = Mag(mag)
        compression_mode = compress

        chunk_size, chunks_per_shard = _get_sharding_parameters(
            chunk_size=chunk_size,
            chunks_per_shard=chunks_per_shard,
            block_len=block_len,
            file_len=file_len,
        )
        if chunk_size is None:
            chunk_size = DEFAULT_CHUNK_SIZE
        if chunks_per_shard is None:
            if self.data_format == DataFormat.Zarr:
                chunks_per_shard = DEFAULT_CHUNKS_PER_SHARD_ZARR
            else:
                chunks_per_shard = DEFAULT_CHUNKS_PER_SHARD

        if chunk_size not in (Vec3Int.full(32), Vec3Int.full(64)):
            warnings.warn(
                "[INFO] `chunk_size` of `32, 32, 32` or `64, 64, 64` is recommended for optimal "
                + f"performance in webKnossos. Got {chunk_size}."
            )

        self._assert_mag_does_not_exist_yet(mag)
        self._create_dir_for_mag(mag)

        mag_view = MagView(
            self,
            mag,
            chunk_size=chunk_size,
            chunks_per_shard=chunks_per_shard,
            compression_mode=compression_mode,
            create=True,
        )

        mag_view._array.ensure_size(
            self.bounding_box.align_with_mag(mag).in_mag(mag).bottomright
        )

        self._mags[mag] = mag_view
        mag_array_info = mag_view.info
        self._properties.mags += [
            MagViewProperties(
                mag=Mag(mag_view.name),
                cube_length=(
                    mag_array_info.shard_size.x
                    if mag_array_info.data_format == DataFormat.WKW
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
                    mag_array_info.shard_size.x
                    if mag_array_info.data_format == DataFormat.WKW
                    else None
                ),
            )
        )
        self.dataset._export_as_json()

        return mag_view

    def get_or_add_mag(
        self,
        mag: Union[int, str, list, tuple, np.ndarray, Mag],
        chunk_size: Optional[Union[Vec3IntLike, int]] = None,
        chunks_per_shard: Optional[Union[Vec3IntLike, int]] = None,
        compress: Optional[bool] = None,
        block_len: Optional[int] = None,  # deprecated
        file_len: Optional[int] = None,  # deprecated
    ) -> MagView:
        """
        Creates a new mag called and adds it to the dataset, in case it did not exist before.
        Then, returns the mag.

        See `add_mag` for more information.
        """

        # normalize the name of the mag
        mag = Mag(mag)
        compression_mode = compress

        chunk_size, chunks_per_shard = _get_sharding_parameters(
            chunk_size=chunk_size,
            chunks_per_shard=chunks_per_shard,
            block_len=block_len,
            file_len=file_len,
        )

        if mag in self._mags.keys():
            assert (
                chunk_size is None or self._mags[mag].info.chunk_size == chunk_size
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
                chunk_size=chunk_size,
                chunks_per_shard=chunks_per_shard,
                compress=compression_mode or False,
            )

    def delete_mag(self, mag: Union[int, str, list, tuple, np.ndarray, Mag]) -> None:
        """
        Deletes the MagView from the `datasource-properties.json` and the data from disk.

        This function raises an `IndexError` if the specified `mag` does not exist.
        """
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

    def _add_foreign_mag(
        self,
        foreign_mag_view_or_path: Union[PathLike, str, MagView],
        symlink: bool,
        make_relative: bool,
        extend_layer_bounding_box: bool = True,
    ) -> MagView:
        """
        The foreign mag is (shallow) copied and the existing mag is added to the datasource-properties.json.
        If extend_layer_bounding_box is true, the self.bounding_box will be extended
        by the bounding box of the layer the foreign mag belongs to.
        Symlinked mags can only be added to layers on local file systems.
        """

        if isinstance(foreign_mag_view_or_path, MagView):
            foreign_mag_view = foreign_mag_view_or_path
        else:
            # local import to prevent circular dependency
            from .dataset import Dataset

            foreign_mag_view_path = make_upath(foreign_mag_view_or_path)
            foreign_mag_view = (
                Dataset.open(foreign_mag_view_path.parent.parent)
                .get_layer(foreign_mag_view_path.parent.name)
                .get_mag(foreign_mag_view_path.name)
            )

        self._assert_mag_does_not_exist_yet(foreign_mag_view.mag)

        if symlink:
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

            (self.path / str(foreign_mag_view.mag)).symlink_to(
                foreign_normalized_mag_path
            )
        else:
            copytree(
                foreign_mag_view.path,
                self.path / str(foreign_mag_view.mag),
            )

        self.add_mag_for_existing_files(foreign_mag_view.mag)
        if extend_layer_bounding_box:
            self.bounding_box = self.bounding_box.extended_by(
                foreign_mag_view.layer.bounding_box
            )
        self.dataset._export_as_json()

        return self._mags[foreign_mag_view.mag]

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
        return self._add_foreign_mag(
            foreign_mag_view_or_path,
            symlink=True,
            make_relative=make_relative,
            extend_layer_bounding_box=extend_layer_bounding_box,
        )

    def add_copy_mag(
        self,
        foreign_mag_view_or_path: Union[PathLike, str, MagView],
        extend_layer_bounding_box: bool = True,
    ) -> MagView:
        """
        Copies the data at `foreign_mag_view_or_path` which belongs to another dataset to the current dataset.
        Additionally, the relevant information from the `datasource-properties.json` of the other dataset are copied too.
        """
        return self._add_foreign_mag(
            foreign_mag_view_or_path,
            symlink=False,
            make_relative=False,
            extend_layer_bounding_box=extend_layer_bounding_box,
        )

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

    @property
    def bounding_box(self) -> BoundingBox:
        return self._properties.bounding_box

    @bounding_box.setter
    def bounding_box(self, bbox: BoundingBox) -> None:
        """
        Updates the offset and size of the bounding box of this layer in the properties.
        """
        assert (
            bbox.topleft.is_positive()
        ), f"Updating the bounding box of layer {self} to {bbox} failed, topleft must not contain negative dimensions."
        self._properties.bounding_box = bbox
        self.dataset._export_as_json()
        for mag in self.mags.values():
            mag._array.ensure_size(
                bbox.align_with_mag(mag.mag).in_mag(mag.mag).bottomright
            )

    def downsample(
        self,
        from_mag: Optional[Mag] = None,
        max_mag: Optional[Mag] = None,
        interpolation_mode: str = "default",
        compress: bool = True,
        sampling_mode: str = SamplingModes.ANISOTROPIC,
        buffer_shape: Optional[Vec3Int] = None,
        force_sampling_scheme: bool = False,
        args: Optional[Namespace] = None,
        allow_overwrite: bool = False,
        only_setup_mags: bool = False,
    ) -> None:
        """
        Downsamples the data starting from `from_mag` until a magnification is `>= max(max_mag)`.
        There are three different `sampling_modes`:
        - 'anisotropic' - The next magnification is chosen so that the width, height and depth of a downsampled voxel assimilate. For example, if the z resolution is worse than the x/y resolution, z won't be downsampled in the first downsampling step(s). As a basis for this method, the scale from the datasource-properties.json is used.
        - 'isotropic' - Each dimension is downsampled equally.
        - 'constant_z' - The x and y dimensions are downsampled equally, but the z dimension remains the same.

        See `downsample_mag` for more information.

        Example:
        ```python
        from webknossos.dataset.downsampling_utils import SamplingModes

        # ...
        # let 'layer' be a `Layer` with only `Mag(1)`
        assert "1" in self.mags.keys()

        layer.downsample(
            max_mag=Mag(4),
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

        if max_mag is None:
            max_mag = calculate_default_max_mag(self.bounding_box.size)

        if self._properties.bounding_box.size.z == 1:
            if sampling_mode != SamplingModes.CONSTANT_Z:
                warnings.warn(
                    "The sampling_mode was changed to 'CONSTANT_Z'. Downsampling 2D data with a different sampling mode mixes in black and thus leads to darkened images."
                )
                sampling_mode = SamplingModes.CONSTANT_Z

        scale: Optional[Tuple[float, float, float]] = None
        if sampling_mode == SamplingModes.ANISOTROPIC or sampling_mode == "auto":
            scale = self.dataset.scale
        elif sampling_mode == SamplingModes.ISOTROPIC:
            scale = None
        elif sampling_mode == SamplingModes.CONSTANT_Z:
            max_mag_with_fixed_z = max_mag.to_list()
            max_mag_with_fixed_z[2] = from_mag.to_list()[2]
            max_mag = Mag(max_mag_with_fixed_z)
            scale = None
        else:
            raise AttributeError(
                f"Downsampling failed: {sampling_mode} is not a valid SamplingMode ({SamplingModes.ANISOTROPIC}, {SamplingModes.ISOTROPIC}, {SamplingModes.CONSTANT_Z})"
            )

        mags_to_downsample = calculate_mags_to_downsample(from_mag, max_mag, scale)

        if len(set([max(m.to_list()) for m in mags_to_downsample])) != len(
            mags_to_downsample
        ):
            msg = (
                f"The downsampling scheme contains multiple magnifications with the same maximum value. This is not supported by webknossos. "
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
            )

    def downsample_mag(
        self,
        from_mag: Mag,
        target_mag: Mag,
        interpolation_mode: str = "default",
        compress: bool = True,
        buffer_shape: Optional[Vec3Int] = None,
        args: Optional[Namespace] = None,
        allow_overwrite: bool = False,
        only_setup_mag: bool = False,
    ) -> None:
        """
        Performs a single downsampling step from `from_mag` to `target_mag`.

        The supported `interpolation_modes` are:
         - "median"
         - "mode"
         - "nearest"
         - "bilinear"
         - "bicubic"

        The `args` can contain information to distribute the computation.
        If allow_overwrite is True, an existing Mag may be overwritten.

        If only_setup_mag is True, the magnification is created, but left
        empty. This parameter can be used to prepare for parallel downsampling
        of multiple layers while avoiding parallel writes with outdated updates
        to the datasource-properties.json file.
        """
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
        target_view = target_mag_view.get_view(
            absolute_offset=bb_mag1.topleft,
            size=bb_mag1.size,
        )

        source_view = prev_mag_view.get_view(
            absolute_offset=bb_mag1.topleft,
            size=bb_mag1.size,
            read_only=True,
        )

        # perform downsampling
        with get_executor_for_args(args) as executor:
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
        args: Optional[Namespace] = None,
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
        )

    def downsample_mag_list(
        self,
        from_mag: Mag,
        target_mags: List[Mag],
        interpolation_mode: str = "default",
        compress: bool = True,
        buffer_shape: Optional[Vec3Int] = None,
        args: Optional[Namespace] = None,
        allow_overwrite: bool = False,
        only_setup_mags: bool = False,
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
            )
            source_mag = target_mag

    def upsample(
        self,
        from_mag: Mag,
        min_mag: Optional[Mag],
        compress: bool,
        sampling_mode: str = SamplingModes.ANISOTROPIC,
        buffer_shape: Optional[Vec3Int] = None,
        buffer_edge_len: Optional[int] = None,
        args: Optional[Namespace] = None,
    ) -> None:
        """
        Upsamples the data starting from `from_mag` as long as the magnification is `>= min_mag`.
        There are three different `sampling_modes`:
        - 'anisotropic' - The next magnification is chosen so that the width, height and depth of a downsampled voxel assimilate. For example, if the z resolution is worse than the x/y resolution, z won't be downsampled in the first downsampling step(s). As a basis for this method, the scale from the datasource-properties.json is used.
        - 'isotropic' - Each dimension is downsampled equally.
        - 'constant_z' - The x and y dimensions are downsampled equally, but the z dimension remains the same.
        """
        assert (
            from_mag in self.mags.keys()
        ), f"Failed to downsample data. The from_mag ({from_mag.to_layer_name()}) does not exist."

        if min_mag is None:
            min_mag = Mag(1)

        scale: Optional[Tuple[float, float, float]] = None
        if sampling_mode == SamplingModes.ANISOTROPIC or sampling_mode == "auto":
            scale = self.dataset.scale
        elif sampling_mode == SamplingModes.ISOTROPIC:
            scale = None
        elif sampling_mode == SamplingModes.CONSTANT_Z:
            min_mag_with_fixed_z = min_mag.to_list()
            min_mag_with_fixed_z[2] = from_mag.to_list()[2]
            min_mag = Mag(min_mag_with_fixed_z)
            scale = self.dataset.scale
        else:
            raise AttributeError(
                f"Upsampling failed: {sampling_mode} is not a valid UpsamplingMode ({SamplingModes.ANISOTROPIC}, {SamplingModes.ISOTROPIC}, {SamplingModes.CONSTANT_Z})"
            )

        if buffer_shape is None and buffer_edge_len is not None:
            buffer_shape = Vec3Int.full(buffer_edge_len)

        mags_to_upsample = calculate_mags_to_upsample(from_mag, min_mag, scale)

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

            # Get target view
            target_view = target_mag_view.get_view()

            # perform upsampling
            with get_executor_for_args(args) as executor:

                if buffer_shape is None:
                    buffer_shape = determine_buffer_shape(prev_mag_view.info)
                func = named_partial(
                    upsample_cube_job,
                    mag_factors=mag_factors,
                    buffer_shape=buffer_shape,
                )
                prev_mag_view.get_view().for_zipped_chunks(
                    # this view is restricted to the bounding box specified in the properties
                    func,
                    target_view=target_view,
                    executor=executor,
                    progress_desc=f"Upsampling from Mag {prev_mag} to Mag {target_mag}",
                )

    def _setup_mag(self, mag: Mag) -> None:
        # This method is used to initialize the mag when opening the Dataset. This does not create e.g. the wk_header.

        mag_name = mag.to_layer_name()

        self._assert_mag_does_not_exist_yet(mag)

        try:
            cls_array = BaseArray.get_class(self._properties.data_format)
            info = cls_array.open(
                _find_mag_path_on_disk(self.dataset.path, self.name, mag_name)
            ).info
            self._mags[mag] = MagView(
                self,
                mag,
                info.chunk_size,
                info.chunks_per_shard,
                info.compression_mode,
            )
        except ArrayException as e:
            logging.error(
                f"Failed to setup magnification {mag_name}, which is specified in the datasource-properties.json. See {e}"
            )

    def _initialize_mag_from_other_mag(
        self, new_mag_name: Union[str, Mag], other_mag: MagView, compress: bool
    ) -> MagView:
        return self.add_mag(
            new_mag_name,
            chunk_size=other_mag.info.chunk_size,
            chunks_per_shard=other_mag.info.chunks_per_shard,
            compress=compress,
        )

    def __repr__(self) -> str:
        return repr(
            "Layer(%s, dtype_per_channel=%s, num_channels=%s)"
            % (self.name, self.dtype_per_channel, self.num_channels)
        )

    @property
    def category(self) -> LayerCategoryType:
        return COLOR_CATEGORY

    @property
    def dtype_per_layer(self) -> str:
        return _dtype_per_channel_to_dtype_per_layer(
            self.dtype_per_channel, self.num_channels
        )

    def _get_largest_segment_id_maybe(self) -> Optional[int]:
        return None


class SegmentationLayer(Layer):

    _properties: SegmentationLayerProperties

    @property
    def largest_segment_id(self) -> int:
        return self._properties.largest_segment_id

    @largest_segment_id.setter
    def largest_segment_id(self, largest_segment_id: int) -> None:
        if type(largest_segment_id) != int:
            assert largest_segment_id == int(
                largest_segment_id
            ), f"A non-integer value was passed for largest_segment_id ({largest_segment_id})."
            largest_segment_id = int(largest_segment_id)

        self._properties.largest_segment_id = largest_segment_id
        self.dataset._export_as_json()

    @property
    def category(self) -> LayerCategoryType:
        return SEGMENTATION_CATEGORY

    def _get_largest_segment_id_maybe(self) -> Optional[int]:
        return self.largest_segment_id
