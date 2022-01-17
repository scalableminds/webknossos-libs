import logging
import operator
import os
import re
import shutil
import warnings
from argparse import Namespace
from os import makedirs
from os.path import join
from pathlib import Path
from shutil import rmtree
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np
from wkw import wkw

from webknossos.geometry import BoundingBox, Mag

from .downsampling_utils import (
    SamplingModes,
    calculate_default_max_mag,
    calculate_mags_to_downsample,
    calculate_mags_to_upsample,
    determine_buffer_edge_len,
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

from webknossos.utils import get_executor_for_args, named_partial

from .defaults import DEFAULT_WKW_FILE_LEN
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


class Layer:
    """
    A `Layer` consists of multiple `webknossos.dataset.mag_view.MagView`s, which store the same data in different magnifications.
    """

    def __init__(self, dataset: "Dataset", properties: LayerProperties) -> None:
        """
        Do not use this constructor manually. Instead use `webknossos.dataset.layer.Dataset.add_layer()` to create a `Layer`.
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

        makedirs(self.path, exist_ok=True)

        for resolution in properties.wkw_resolutions:
            self._setup_mag(Mag(resolution.resolution))
        # Only keep the properties of resolutions that were initialized.
        # Sometimes the directory of a resolution is removed from disk manually, but the properties are not updated.
        self._properties.wkw_resolutions = [
            res
            for res in self._properties.wkw_resolutions
            if Mag(res.resolution) in self._mags
        ]

    @property
    def path(self) -> Path:
        return Path(join(self.dataset.path, self.name))

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
        """
        assert (
            layer_name not in self.dataset.layers.keys()
        ), f"Failed to rename layer {self.name} to {layer_name}: The new name already exists."
        os.rename(self.dataset.path / self.name, self.dataset.path / layer_name)
        del self.dataset.layers[self.name]
        self.dataset.layers[layer_name] = self
        self._properties.name = layer_name
        self._name = layer_name

        # The MagViews need to be updated
        for mag in self._mags.values():
            mag._path = _find_mag_path_on_disk(self.dataset.path, self.name, mag.name)
            # Deleting the dataset will close the file handle.
            # The new dataset will be opened automatically when needed.
            del mag._wkw_dataset

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
        block_len: int = 32,
        file_len: int = DEFAULT_WKW_FILE_LEN,
        compress: bool = False,
    ) -> MagView:
        """
        Creates a new mag called and adds it to the layer.
        The parameter `block_len`, `file_len` and `compress` can be
        specified to adjust how the data is stored on disk.
        Note that writing compressed data which is not aligned with the blocks on disk may result in
        diminished performance, as full blocks will automatically be read to pad the write actions. Alternatively,
        you can call mag.compress() after all the data was written

        The return type is `webknossos.dataset.mag_view.MagView`.

        Raises an IndexError if the specified `mag` already exists.
        """
        # normalize the name of the mag
        mag = Mag(mag)
        block_type = (
            wkw.Header.BLOCK_TYPE_LZ4HC if compress else wkw.Header.BLOCK_TYPE_RAW
        )

        self._assert_mag_does_not_exist_yet(mag)
        self._create_dir_for_mag(mag)

        mag_view = MagView(self, mag, block_len, file_len, block_type, create=True)

        self._mags[mag] = mag_view
        self._properties.wkw_resolutions += [
            MagViewProperties(
                Mag(mag_view.name), mag_view.header.block_len * mag_view.header.file_len
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
        cube_length = mag_view.header.file_len * mag_view.header.block_len
        self._properties.wkw_resolutions.append(
            MagViewProperties(resolution=mag, cube_length=cube_length)
        )
        self.dataset._export_as_json()

        return mag_view

    def get_or_add_mag(
        self,
        mag: Union[int, str, list, tuple, np.ndarray, Mag],
        block_len: int = 32,
        file_len: int = DEFAULT_WKW_FILE_LEN,
        compress: bool = False,
    ) -> MagView:
        """
        Creates a new mag called and adds it to the dataset, in case it did not exist before.
        Then, returns the mag.

        See `add_mag` for more information.
        """

        # normalize the name of the mag
        mag = Mag(mag)
        block_type = (
            wkw.Header.BLOCK_TYPE_LZ4HC if compress else wkw.Header.BLOCK_TYPE_RAW
        )

        if mag in self._mags.keys():
            assert (
                block_len is None or self._mags[mag].header.block_len == block_len
            ), f"Cannot get_or_add_mag: The mag {mag} already exists, but the block lengths do not match"
            assert (
                file_len is None or self._mags[mag].header.file_len == file_len
            ), f"Cannot get_or_add_mag: The mag {mag} already exists, but the file lengths do not match"
            assert (
                block_type is None or self._mags[mag].header.block_type == block_type
            ), f"Cannot get_or_add_mag: The mag {mag} already exists, but the block types do not match"
            return self.get_mag(mag)
        else:
            return self.add_mag(
                mag, block_len=block_len, file_len=file_len, compress=compress
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
        self._properties.wkw_resolutions = [
            res
            for res in self._properties.wkw_resolutions
            if Mag(res.resolution) != mag
        ]
        self.dataset._export_as_json()
        # delete files on disk
        full_path = _find_mag_path_on_disk(
            self.dataset.path, self.name, mag.to_layer_name()
        )
        rmtree(full_path)

    def _add_foreign_mag(
        self,
        foreign_mag_view_or_path: Union[os.PathLike, str, MagView],
        symlink: bool,
        make_relative: bool,
        extend_layer_bounding_box: bool = True,
    ) -> MagView:
        """
        The foreign mag is (shallow) copied and the existing mag is added to the datasource-properties.json.
        If extend_layer_bounding_box is true, the self.bounding_box will be extended
        by the bounding box of the layer the foreign mag belongs to.
        """

        if isinstance(foreign_mag_view_or_path, MagView):
            foreign_mag_view = foreign_mag_view_or_path
        else:
            # local import to prevent circular dependency
            from .dataset import Dataset

            foreign_mag_view_path = Path(foreign_mag_view_or_path)
            foreign_mag_view = (
                Dataset.open(foreign_mag_view_path.parent.parent)
                .get_layer(foreign_mag_view_path.parent.name)
                .get_mag(foreign_mag_view_path.name)
            )

        self._assert_mag_does_not_exist_yet(foreign_mag_view.mag)

        foreign_normalized_mag_path = (
            Path(os.path.relpath(foreign_mag_view.path, self.path))
            if make_relative
            else foreign_mag_view.path.resolve()
        )

        if symlink:
            os.symlink(
                foreign_normalized_mag_path,
                join(self.dataset.path, self.name, str(foreign_mag_view.mag)),
            )
        else:
            shutil.copytree(
                foreign_normalized_mag_path,
                join(self.dataset.path, self.name, str(foreign_mag_view.mag)),
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
        foreign_mag_view_or_path: Union[os.PathLike, str, MagView],
        make_relative: bool = False,
        extend_layer_bounding_box: bool = True,
    ) -> MagView:
        """
        Creates a symlink to the data at `foreign_mag_view_or_path` which belongs to another dataset.
        The relevant information from the `datasource-properties.json` of the other dataset is copied to this dataset.
        Note: If the other dataset modifies its bounding box afterwards, the change does not affect this properties
        (or vice versa).
        If make_relative is True, the symlink is made relative to the current dataset path.
        """
        return self._add_foreign_mag(
            foreign_mag_view_or_path,
            symlink=True,
            make_relative=make_relative,
            extend_layer_bounding_box=extend_layer_bounding_box,
        )

    def add_copy_mag(
        self,
        foreign_mag_view_or_path: Union[os.PathLike, str, MagView],
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
        full_path = join(self.dataset.path, self.name, mag)
        makedirs(full_path, exist_ok=True)

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
        self._properties.bounding_box = bbox
        self.dataset._export_as_json()

    def downsample(
        self,
        from_mag: Optional[Mag] = None,
        max_mag: Optional[Mag] = None,
        interpolation_mode: str = "default",
        compress: bool = True,
        sampling_mode: str = SamplingModes.ANISOTROPIC,
        buffer_edge_len: Optional[int] = None,
        force_sampling_scheme: bool = False,
        args: Optional[Namespace] = None,
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
                buffer_edge_len=buffer_edge_len,
                args=args,
            )

    def downsample_mag(
        self,
        from_mag: Mag,
        target_mag: Mag,
        interpolation_mode: str = "default",
        compress: bool = True,
        buffer_edge_len: Optional[int] = None,
        args: Optional[Namespace] = None,
        allow_overwrite: bool = False,
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

        bb_mag1 = self.bounding_box

        aligned_source_bb = bb_mag1.align_with_mag(target_mag, ceil=True).in_mag(
            from_mag
        )
        aligned_target_bb = bb_mag1.align_with_mag(target_mag, ceil=True).in_mag(
            target_mag
        )

        # Get target view
        target_view = target_mag_view.get_view(
            offset=aligned_target_bb.topleft,
            size=aligned_target_bb.size,
        )

        source_view = prev_mag_view.get_view(
            offset=aligned_source_bb.topleft,
            size=aligned_source_bb.size,
            read_only=True,
        )

        # perform downsampling
        with get_executor_for_args(args) as executor:
            if buffer_edge_len is None:
                buffer_edge_len = determine_buffer_edge_len(
                    prev_mag_view
                )  # DEFAULT_EDGE_LEN
            func = named_partial(
                downsample_cube_job,
                mag_factors=mag_factors,
                interpolation_mode=parsed_interpolation_mode,
                buffer_edge_len=buffer_edge_len,
            )

            source_view.for_zipped_chunks(
                # this view is restricted to the bounding box specified in the properties
                func,
                target_view=target_view,
                source_chunk_size=target_mag_view._get_file_dimensions() * mag_factors,
                target_chunk_size=target_mag_view._get_file_dimensions(),
                executor=executor,
                progress_desc=f"Downsampling layer {self.name} from Mag {from_mag} to Mag {target_mag}",
            )

    def redownsample(
        self,
        interpolation_mode: str = "default",
        compress: bool = True,
        buffer_edge_len: Optional[int] = None,
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
            from_mag,
            target_mags,
            interpolation_mode,
            compress,
            buffer_edge_len,
            args,
            allow_overwrite=True,
        )

    def downsample_mag_list(
        self,
        from_mag: Mag,
        target_mags: List[Mag],
        interpolation_mode: str = "default",
        compress: bool = True,
        buffer_edge_len: Optional[int] = None,
        args: Optional[Namespace] = None,
        allow_overwrite: bool = False,
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
                buffer_edge_len=buffer_edge_len,
                args=args,
                allow_overwrite=allow_overwrite,
            )
            source_mag = target_mag

    def upsample(
        self,
        from_mag: Mag,
        min_mag: Optional[Mag],
        compress: bool,
        sampling_mode: str = SamplingModes.ANISOTROPIC,
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

                if buffer_edge_len is None:
                    buffer_edge_len = determine_buffer_edge_len(
                        prev_mag_view
                    )  # DEFAULT_EDGE_LEN
                func = named_partial(
                    upsample_cube_job,
                    mag_factors=mag_factors,
                    buffer_edge_len=buffer_edge_len,
                )
                prev_mag_view.get_view().for_zipped_chunks(
                    # this view is restricted to the bounding box specified in the properties
                    func,
                    target_view=target_view,
                    source_chunk_size=target_mag_view._get_file_dimensions(),
                    target_chunk_size=target_mag_view._get_file_dimensions()
                    * np.array([int(1 / f) for f in mag_factors]),
                    executor=executor,
                    progress_desc=f"Upsampling from Mag {prev_mag} to Mag {target_mag}",
                )

    def _setup_mag(self, mag: Mag) -> None:
        # This method is used to initialize the mag when opening the Dataset. This does not create e.g. the wk_header.

        mag_name = mag.to_layer_name()

        self._assert_mag_does_not_exist_yet(mag)

        try:
            with wkw.Dataset.open(
                str(_find_mag_path_on_disk(self.dataset.path, self.name, mag_name))
            ) as wkw_dataset:
                wk_header = wkw_dataset.header

            self._mags[mag] = MagView(
                self,
                mag,
                wk_header.block_len,
                wk_header.file_len,
                wk_header.block_type,
            )
        except wkw.WKWException as e:
            logging.error(
                f"Failed to setup magnification {mag_name}, which is specified in the datasource-properties.json. See {e}"
            )

    def _initialize_mag_from_other_mag(
        self, new_mag_name: Union[str, Mag], other_mag: MagView, compress: bool
    ) -> MagView:
        return self.add_mag(
            new_mag_name,
            block_len=other_mag.header.block_len,
            file_len=other_mag.header.file_len,
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


class SegmentationLayer(Layer):

    _properties: SegmentationLayerProperties

    @property
    def largest_segment_id(self) -> int:
        return self._properties.largest_segment_id

    @largest_segment_id.setter
    def largest_segment_id(self, largest_segment_id: int) -> None:
        self._properties.largest_segment_id = largest_segment_id
        self.dataset._export_as_json()

    @property
    def category(self) -> LayerCategoryType:
        return SEGMENTATION_CATEGORY
