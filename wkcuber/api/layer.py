import logging
import math
import os
from argparse import Namespace
from shutil import rmtree
from os.path import join
from os import makedirs
from typing import (
    Tuple,
    Union,
    Dict,
    TYPE_CHECKING,
    cast,
    Optional,
    List,
)

import numpy as np

from wkw import wkw

from wkcuber.api.bounding_box import BoundingBox
from wkcuber.api.properties.layer_properties import SegmentationLayerProperties
from wkcuber.downsampling_utils import (
    calculate_virtual_scale_for_target_mag,
    calculate_default_max_mag,
    get_previous_mag,
    SamplingModes,
)

if TYPE_CHECKING:
    from wkcuber.api.dataset import Dataset
from wkcuber.api.mag_view import (
    MagView,
    _find_mag_path_on_disk,
)
from wkcuber.downsampling_utils import (
    get_next_mag,
    parse_interpolation_mode,
    downsample_cube_job,
    determine_buffer_edge_len,
)
from wkcuber.upsampling_utils import upsample_cube_job
from wkcuber.mag import Mag
from wkcuber.utils import (
    DEFAULT_WKW_FILE_LEN,
    get_executor_for_args,
    named_partial,
    _snake_to_camel_case,
)


class Layer:
    """
    A `Layer` consists of multiple `wkcuber.api.mag_view.MagView`s, which store the same data in different magnifications.

    ## Examples

    ### Adding layer to dataset
    ```python
    from wkcuber.api.dataset import Dataset

    dataset = Dataset(<path_to_dataset>)
    # Adds a new layer
    layer = dataset.get_layer("color")
    ```

    ## Functions
    """

    def __init__(
        self,
        name: str,
        dataset: "Dataset",
        dtype_per_channel: np.dtype,
        num_channels: int,
    ) -> None:
        """
        Do not use this constructor manually. Instead use `wkcuber.api.layer.Dataset.add_layer()` to create a `Layer`.
        """
        self.name = name
        self.dataset = dataset
        self.dtype_per_channel = dtype_per_channel
        self.num_channels = num_channels
        self._mags: Dict[Mag, MagView] = {}

        full_path = join(dataset.path, name)
        makedirs(full_path, exist_ok=True)

    @property
    def mags(self) -> Dict[Mag, MagView]:
        """
        Getter for dictionary containing all mags.
        """
        return self._mags

    def get_mag(self, mag: Union[int, str, list, tuple, np.ndarray, Mag]) -> MagView:
        """
        Returns the MagDataset called `mag` of this layer. The return type is `wkcuber.api.MagDataset`.

        This function raises an `IndexError` if the specified `mag` does not exist.
        """
        mag = Mag(mag)
        if mag not in self.mags.keys():
            raise IndexError(
                "The mag {} is not a mag of this layer".format(mag.to_layer_name())
            )
        return self.mags[mag]

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

        The return type is `wkcuber.api.mag_view.MagView`.

        Raises an IndexError if the specified `mag` already exists.
        """
        # normalize the name of the mag
        mag = Mag(mag)
        block_type = (
            wkw.Header.BLOCK_TYPE_LZ4HC if compress else wkw.Header.BLOCK_TYPE_RAW
        )

        self._assert_mag_does_not_exist_yet(mag)
        self._create_dir_for_mag(mag)

        self._mags[mag] = MagView(
            self, mag.to_layer_name(), block_len, file_len, block_type, create=True
        )
        self.dataset.properties._add_mag(
            self.name, mag.to_layer_name(), cube_length=block_len * file_len
        )

        return self._mags[mag]

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
        Deletes the MagDataset from the `datasource-properties.json` and the data from disk.

        This function raises an `IndexError` if the specified `mag` does not exist.
        """
        mag = Mag(mag)
        if mag not in self.mags.keys():
            raise IndexError(
                "Deleting mag {} failed. There is no mag with this name".format(mag)
            )

        del self._mags[mag]
        self.dataset.properties._delete_mag(self.name, mag.to_layer_name())
        # delete files on disk
        full_path = _find_mag_path_on_disk(
            self.dataset.path, self.name, mag.to_layer_name()
        )
        rmtree(full_path)

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

    def set_bounding_box(
        self, offset: Tuple[int, int, int], size: Tuple[int, int, int]
    ) -> None:
        """
        Updates the offset and size of the bounding box of this layer in the properties.
        """
        self.dataset.properties._set_bounding_box_of_layer(self.name, offset, size)
        bounding_box = BoundingBox(offset, size)

        for mag, mag_view in self.mags.items():
            mag_view.size = cast(
                Tuple[int, int, int],
                tuple(
                    bounding_box.align_with_mag(mag, ceil=True).in_mag(mag).bottomright
                ),
            )

    def set_bounding_box_offset(self, offset: Tuple[int, int, int]) -> None:
        """
        Updates the offset of the bounding box of this layer in the properties.
        """
        size: Tuple[int, int, int] = self.dataset.properties.get_bounding_box_of_layer(
            self.name
        )[1]
        self.set_bounding_box(offset, size)

    def set_bounding_box_size(self, size: Tuple[int, int, int]) -> None:
        """
        Updates the size of the bounding box of this layer in the properties.
        """
        offset: Tuple[
            int, int, int
        ] = self.dataset.properties.get_bounding_box_of_layer(self.name)[0]
        self.set_bounding_box(offset, size)

    def get_bounding_box(self) -> BoundingBox:
        return self.dataset.properties.data_layers[self.name].get_bounding_box()

    def rename(self, layer_name: str) -> None:
        """
        Renames the layer to `layer_name`. This changes the name of the directory on disk and updates the properties.
        """
        assert (
            layer_name not in self.dataset.layers.keys()
        ), f"Failed to rename layer {self.name} to {layer_name}: The new name already exists."
        os.rename(self.dataset.path / self.name, self.dataset.path / layer_name)
        layer_properties = self.dataset.properties.data_layers[self.name]
        layer_properties._name = layer_name
        del self.dataset.properties.data_layers[self.name]
        self.dataset.properties._data_layers[layer_name] = layer_properties
        self.dataset.properties._export_as_json()
        del self.dataset.layers[self.name]
        self.dataset.layers[layer_name] = self
        self.name = layer_name

        # The MagViews need to be updated
        for mag in self._mags.values():
            mag.path = _find_mag_path_on_disk(self.dataset.path, self.name, mag.name)
            if mag._is_opened:
                # Reopen handle to dataset on disk
                mag.close()
                mag.open()

    def downsample(
        self,
        from_mag: Optional[Mag] = None,
        max_mag: Optional[Mag] = None,
        interpolation_mode: str = "default",
        compress: bool = True,
        sampling_mode: str = SamplingModes.AUTO,
        buffer_edge_len: Optional[int] = None,
        args: Optional[Namespace] = None,
    ) -> None:
        """
        Downsamples the data starting from `from_mag` until a magnification is `>= max(max_mag)`.
        There are three different `sampling_modes`:
        - 'auto' - The next magnification is chosen so that the width, height and depth of a downsampled voxel assimilate. For example, if the z resolution is worse than the x/y resolution, z won't be downsampled in the first downsampling step(s). As a basis for this method, the scale from the datasource-properties.json is used.
        - 'isotropic' - Each dimension is downsampled equally.
        - 'constant_z' - The x and y dimensions are downsampled equally, but the z dimension remains the same.

        See `downsample_mag` for more information.

        Example:
        ```python
        from wkcuber.downsampling_utils import SamplingModes

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
            max_mag = calculate_default_max_mag(
                self.dataset.properties.data_layers[self.name].get_bounding_box_size()
            )

        if sampling_mode == SamplingModes.AUTO:
            scale = self.dataset.properties.scale
        elif sampling_mode == SamplingModes.ISOTROPIC:
            scale = (1, 1, 1)
        elif sampling_mode == SamplingModes.CONSTANT_Z:
            max_mag_with_fixed_z = max_mag.to_array()
            max_mag_with_fixed_z[2] = from_mag.to_array()[2]
            max_mag = Mag(max_mag_with_fixed_z)
            scale = calculate_virtual_scale_for_target_mag(max_mag)
        else:
            raise AttributeError(
                f"Downsampling failed: {sampling_mode} is not a valid SamplingMode ({SamplingModes.AUTO}, {SamplingModes.ISOTROPIC}, {SamplingModes.CONSTANT_Z})"
            )

        prev_mag = from_mag
        target_mag = get_next_mag(prev_mag, scale)

        while target_mag <= max_mag:
            self.downsample_mag(
                from_mag=prev_mag,
                target_mag=target_mag,
                interpolation_mode=interpolation_mode,
                compress=compress,
                buffer_edge_len=buffer_edge_len,
                args=args,
            )

            prev_mag = target_mag
            target_mag = get_next_mag(target_mag, scale)

    def downsample_mag(
        self,
        from_mag: Mag,
        target_mag: Mag,
        interpolation_mode: str = "default",
        compress: bool = True,
        buffer_edge_len: Optional[int] = None,
        args: Optional[Namespace] = None,
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
        """
        assert (
            from_mag in self.mags.keys()
        ), f"Failed to downsample data. The from_mag ({from_mag.to_layer_name()}) does not exist."

        parsed_interpolation_mode = parse_interpolation_mode(
            interpolation_mode, self.dataset.properties.data_layers[self.name].category
        )

        assert from_mag <= target_mag
        assert target_mag not in self.mags

        prev_mag_view = self.mags[from_mag]

        mag_factors = [
            t // s for (t, s) in zip(target_mag.to_array(), from_mag.to_array())
        ]

        # initialize the new mag
        target_mag_view = self._initialize_mag_from_other_mag(
            target_mag, prev_mag_view, compress
        )

        bb_mag1 = BoundingBox(
            topleft=self.dataset.properties.data_layers[
                self.name
            ].get_bounding_box_offset(),
            size=self.dataset.properties.data_layers[self.name].get_bounding_box_size(),
        )

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
            voxel_count_per_cube = np.prod(prev_mag_view._get_file_dimensions())
            job_count_per_log = math.ceil(
                1024 ** 3 / voxel_count_per_cube
            )  # log every gigavoxel of processed data

            if buffer_edge_len is None:
                buffer_edge_len = determine_buffer_edge_len(
                    prev_mag_view
                )  # DEFAULT_EDGE_LEN
            func = named_partial(
                downsample_cube_job,
                mag_factors=mag_factors,
                interpolation_mode=parsed_interpolation_mode,
                buffer_edge_len=buffer_edge_len,
                job_count_per_log=job_count_per_log,
            )

            source_view.for_zipped_chunks(
                # this view is restricted to the bounding box specified in the properties
                func,
                target_view=target_view,
                source_chunk_size=np.array(target_mag_view._get_file_dimensions())
                * mag_factors,
                target_chunk_size=target_mag_view._get_file_dimensions(),
                executor=executor,
            )

        logging.info("Mag {0} successfully cubed".format(target_mag))

    def downsample_mag_list(
        self,
        from_mag: Mag,
        target_mags: List[Mag],
        interpolation_mode: str = "default",
        compress: bool = True,
        buffer_edge_len: Optional[int] = None,
        args: Optional[Namespace] = None,
    ) -> None:
        """
        Downsamples the data starting at `from_mag` to each magnification in `target_mags` iteratively.

        See `downsample_mag` for more information.
        """
        assert (
            from_mag in self.mags.keys()
        ), f"Failed to downsample data. The from_mag ({from_mag}) does not exist."

        # The lambda function is important because 'sorted(target_mags)' would only sort by the maximum element per mag
        target_mags = sorted(target_mags, key=lambda m: m.to_array())

        for i in range(len(target_mags) - 1):
            assert np.less_equal(
                target_mags[i].as_np(), target_mags[i + 1].as_np()
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
            )
            source_mag = target_mag

    def upsample(
        self,
        from_mag: Mag,
        min_mag: Optional[Mag],
        compress: bool,
        sampling_mode: str = SamplingModes.AUTO,
        buffer_edge_len: Optional[int] = None,
        args: Optional[Namespace] = None,
    ) -> None:
        """
        Upsamples the data starting from `from_mag` as long as the magnification is `>= min_mag`.
        There are three different `sampling_modes`:
        - 'auto' - The next magnification is chosen so that the width, height and depth of a downsampled voxel assimilate. For example, if the z resolution is worse than the x/y resolution, z won't be downsampled in the first downsampling step(s). As a basis for this method, the scale from the datasource-properties.json is used.
        - 'isotropic' - Each dimension is downsampled equally.
        - 'constant_z' - The x and y dimensions are downsampled equally, but the z dimension remains the same.
        """
        assert (
            from_mag in self.mags.keys()
        ), f"Failed to downsample data. The from_mag ({from_mag.to_layer_name()}) does not exist."

        if min_mag is None:
            min_mag = Mag(1)

        if sampling_mode == SamplingModes.AUTO:
            scale = self.dataset.properties.scale
        elif sampling_mode == SamplingModes.ISOTROPIC:
            scale = (1, 1, 1)
        elif sampling_mode == SamplingModes.CONSTANT_Z:
            min_mag_with_fixed_z = min_mag.to_array()
            min_mag_with_fixed_z[2] = from_mag.to_array()[2]
            min_mag = Mag(min_mag_with_fixed_z)
            scale = calculate_virtual_scale_for_target_mag(min_mag)
        else:
            raise AttributeError(
                f"Upsampling failed: {sampling_mode} is not a valid UpsamplingMode ({SamplingModes.AUTO}, {SamplingModes.ISOTROPIC}, {SamplingModes.CONSTANT_Z})"
            )

        prev_mag = from_mag
        target_mag = get_previous_mag(prev_mag, scale)

        while target_mag >= min_mag and prev_mag > Mag(1):
            assert prev_mag > target_mag
            assert target_mag not in self.mags

            prev_mag_view = self.mags[prev_mag]

            mag_factors = [
                t / s for (t, s) in zip(target_mag.to_array(), prev_mag.to_array())
            ]

            # initialize the new mag
            target_mag_view = self._initialize_mag_from_other_mag(
                target_mag, prev_mag_view, compress
            )

            # Get target view
            target_view = target_mag_view.get_view()

            # perform upsampling
            with get_executor_for_args(args) as executor:
                voxel_count_per_cube = np.prod(prev_mag_view._get_file_dimensions())
                job_count_per_log = math.ceil(
                    1024 ** 3 / voxel_count_per_cube
                )  # log every gigavoxel of processed data

                if buffer_edge_len is None:
                    buffer_edge_len = determine_buffer_edge_len(
                        prev_mag_view
                    )  # DEFAULT_EDGE_LEN
                func = named_partial(
                    upsample_cube_job,
                    mag_factors=mag_factors,
                    buffer_edge_len=buffer_edge_len,
                    job_count_per_log=job_count_per_log,
                )
                prev_mag_view.get_view().for_zipped_chunks(
                    # this view is restricted to the bounding box specified in the properties
                    func,
                    target_view=target_view,
                    source_chunk_size=target_mag_view._get_file_dimensions(),
                    target_chunk_size=target_mag_view._get_file_dimensions()
                    * np.array([int(1 / f) for f in mag_factors]),
                    executor=executor,
                )

            logging.info("Mag {0} successfully cubed".format(target_mag))

            prev_mag = target_mag
            target_mag = get_previous_mag(target_mag, scale)

    def _setup_mag(self, mag: Union[str, Mag]) -> None:
        # This method is used to initialize the mag when opening the Dataset. This does not create e.g. the wk_header.

        # normalize the name of the mag
        mag = Mag(mag)
        mag_name = mag.to_layer_name()

        self._assert_mag_does_not_exist_yet(mag)

        try:
            with wkw.Dataset.open(
                str(_find_mag_path_on_disk(self.dataset.path, self.name, mag_name))
            ) as wkw_dataset:
                wk_header = wkw_dataset.header

            self._mags[mag] = MagView(
                self,
                mag_name,
                wk_header.block_len,
                wk_header.file_len,
                wk_header.block_type,
            )

            self.dataset.properties._add_mag(
                self.name,
                mag_name,
                cube_length=wk_header.block_len * wk_header.file_len,
            )
        except wkw.WKWException:
            logging.error(
                f"Failed to setup magnification {mag_name}, which is specified in the datasource-properties.json"
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

    def set_view_configuration(
        self, view_configuration: "LayerViewConfiguration"
    ) -> None:
        self.dataset.properties._data_layers[self.name]._default_view_configuration = {
            _snake_to_camel_case(k): v
            for k, v in vars(view_configuration).items()
            if v is not None
        }
        self.dataset.properties._export_as_json()  # update properties on disk

    def get_view_configuration(self) -> Optional["LayerViewConfiguration"]:
        view_configuration_dict = self.dataset.properties.data_layers[
            self.name
        ].default_view_configuration
        if view_configuration_dict is None:
            return None

        return LayerViewConfiguration(
            color=cast(Tuple[int, int, int], tuple(view_configuration_dict["color"])),
            alpha=view_configuration_dict.get("alpha"),
            intensity_range=cast(
                Tuple[float, float], tuple(view_configuration_dict["intensityRange"])
            )
            if "intensityRange" in view_configuration_dict.keys()
            else None,
            min=view_configuration_dict.get("min"),
            max=view_configuration_dict.get("max"),
            is_disabled=view_configuration_dict.get("isDisabled"),
            is_inverted=view_configuration_dict.get("isInverted"),
            is_in_edit_mode=view_configuration_dict.get("isInEditMode"),
        )

    def __repr__(self) -> str:
        return repr(
            "Layer(%s, dtype_per_channel=%s, num_channels=%s)"
            % (self.name, self.dtype_per_channel, self.num_channels)
        )


class SegmentationLayer(Layer):
    @property
    def largest_segment_id(self) -> int:
        layer_properties = self.dataset.properties.data_layers[self.name]
        assert isinstance(layer_properties, SegmentationLayerProperties)
        return layer_properties.largest_segment_id

    @largest_segment_id.setter
    def largest_segment_id(self, largest_segment_id: int) -> None:
        layer_properties = self.dataset.properties._data_layers[self.name]
        assert isinstance(layer_properties, SegmentationLayerProperties)
        layer_properties._largest_segment_id = largest_segment_id
        self.dataset.properties._export_as_json()


class LayerCategories:
    """
    There are two different types of layers.
    This class can be used to specify the type of a layer during creation:
    ```python
    from wkcuber.api.dataset import Dataset

    dataset = Dataset(<path_to_dataset>)
    # Adds a new layer
    layer = dataset.add_layer("color", LayerCategories.COLOR_TYPE)
    ```
    """

    COLOR_TYPE = "color"
    SEGMENTATION_TYPE = "segmentation"


class LayerViewConfiguration:
    """
    Stores information on how the dataset is shown in webknossos by default.
    """

    def __init__(
        self,
        color: Optional[Tuple[int, int, int]] = None,
        alpha: Optional[float] = None,
        intensity_range: Optional[Tuple[float, float]] = None,
        min: Optional[float] = None,  # pylint: disable=redefined-builtin
        max: Optional[float] = None,  # pylint: disable=redefined-builtin
        is_disabled: Optional[bool] = None,
        is_inverted: Optional[bool] = None,
        is_in_edit_mode: Optional[bool] = None,
    ):
        self.color = color
        self.alpha = alpha
        self.intensity_range = intensity_range
        self.min = min
        self.max = max
        self.is_disabled = is_disabled
        self.is_inverted = is_inverted
        self.is_in_edit_mode = is_in_edit_mode
