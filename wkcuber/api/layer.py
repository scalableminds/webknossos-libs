import logging
import math
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
        self._mags: Dict[str, MagView] = {}

        full_path = join(dataset.path, name)
        makedirs(full_path, exist_ok=True)

    @property
    def mags(self) -> Dict[str, MagView]:
        """
        Getter for dictionary containing all mags.
        """
        return self._mags

    def get_mag(self, mag: Union[int, str, list, tuple, np.ndarray, Mag]) -> MagView:
        """
        Returns the MagDataset called `mag` of this layer. The return type is `wkcuber.api.MagDataset`.

        This function raises an `IndexError` if the specified `mag` does not exist.
        """
        mag = Mag(mag).to_layer_name()
        if mag not in self.mags.keys():
            raise IndexError("The mag {} is not a mag of this layer".format(mag))
        return self.mags[mag]

    def add_mag(
        self,
        mag: Union[int, str, list, tuple, np.ndarray, Mag],
        block_len: int = 32,
        file_len: int = DEFAULT_WKW_FILE_LEN,
        block_type: int = wkw.Header.BLOCK_TYPE_RAW,
    ) -> MagView:
        """
        Creates a new mag called and adds it to the layer.
        The parameter `block_len`, `file_len` and `block_type` can be
        specified to adjust how the data is stored on disk.

        The return type is `wkcuber.api.mag_view.MagView`.

        Raises an IndexError if the specified `mag` already exists.
        """
        # normalize the name of the mag
        mag = Mag(mag).to_layer_name()

        self._assert_mag_does_not_exist_yet(mag)
        self._create_dir_for_mag(mag)

        self._mags[mag] = MagView(
            self, mag, block_len, file_len, block_type, create=True
        )
        self.dataset.properties._add_mag(
            self.name, mag, cube_length=block_len * file_len
        )

        return self._mags[mag]

    def get_or_add_mag(
        self,
        mag: Union[int, str, list, tuple, np.ndarray, Mag],
        block_len: int = 32,
        file_len: int = DEFAULT_WKW_FILE_LEN,
        block_type: int = wkw.Header.BLOCK_TYPE_RAW,
    ) -> MagView:
        """
        Creates a new mag called and adds it to the dataset, in case it did not exist before.
        Then, returns the mag.

        See `add_mag` for more information.
        """

        # normalize the name of the mag
        mag = Mag(mag).to_layer_name()

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
                mag, block_len=block_len, file_len=file_len, block_type=block_type
            )

    def delete_mag(self, mag: Union[int, str, list, tuple, np.ndarray, Mag]) -> None:
        """
        Deletes the MagDataset from the `datasource-properties.json` and the data from disk.

        This function raises an `IndexError` if the specified `mag` does not exist.
        """
        mag = Mag(mag).to_layer_name()
        if mag not in self.mags.keys():
            raise IndexError(
                "Deleting mag {} failed. There is no mag with this name".format(mag)
            )

        del self._mags[mag]
        self.dataset.properties._delete_mag(self.name, mag)
        # delete files on disk
        full_path = _find_mag_path_on_disk(self.dataset.path, self.name, mag)
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
        mag = Mag(mag).to_layer_name()
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

        for mag_name, mag in self.mags.items():
            mag.size = cast(
                Tuple[int, int, int],
                tuple(
                    bounding_box.align_with_mag(Mag(mag_name), ceil=True)
                    .in_mag(Mag(mag_name))
                    .bottomright
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
            from_mag = Mag(max(self.mags.keys()))

        assert (
            from_mag.to_layer_name() in self.mags.keys()
        ), f"Failed to downsample data. The from_mag ({from_mag}) does not exist."

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
            from_mag.to_layer_name() in self.mags.keys()
        ), f"Failed to downsample data. The from_mag ({from_mag}) does not exist."

        parsed_interpolation_mode = parse_interpolation_mode(
            interpolation_mode, self.dataset.properties.data_layers[self.name].category
        )

        assert from_mag <= target_mag
        assert target_mag.to_layer_name() not in self.mags

        prev_mag_view = self.mags[from_mag.to_layer_name()]

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
                compress=compress,
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
            from_mag.to_layer_name() in self.mags.keys()
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
            from_mag.to_layer_name() in self.mags.keys()
        ), f"Failed to downsample data. The from_mag ({from_mag}) does not exist."

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
            assert target_mag.to_layer_name() not in self.mags

            prev_mag_view = self.mags[prev_mag.to_layer_name()]

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
                    compress=compress,
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
        mag = Mag(mag).to_layer_name()

        self._assert_mag_does_not_exist_yet(mag)

        try:
            with wkw.Dataset.open(
                str(_find_mag_path_on_disk(self.dataset.path, self.name, mag))
            ) as wkw_dataset:
                wk_header = wkw_dataset.header

            self._mags[mag] = MagView(
                self, mag, wk_header.block_len, wk_header.file_len, wk_header.block_type
            )

            self.dataset.properties._add_mag(
                self.name, mag, cube_length=wk_header.block_len * wk_header.file_len
            )
        except wkw.WKWException:
            logging.error(
                f"Failed to setup magnification {str(mag)}, which is specified in the datasource-properties.json"
            )

    def _initialize_mag_from_other_mag(
        self, new_mag_name: Union[str, Mag], other_mag: MagView, compress: bool
    ) -> MagView:
        block_type = (
            wkw.Header.BLOCK_TYPE_LZ4HC if compress else wkw.Header.BLOCK_TYPE_RAW
        )
        return self.add_mag(
            new_mag_name,
            block_len=other_mag.header.block_len,
            file_len=other_mag.header.file_len,
            block_type=block_type,
        )


class LayerTypes:
    """
    There are two different types of layers.
    This class can be used to specify the type of a layer during creation:
    ```python
    from wkcuber.api.dataset import Dataset

    dataset = Dataset(<path_to_dataset>)
    # Adds a new layer
    layer = dataset.add_layer("color", LayerTypes.COLOR_TYPE)
    ```
    """

    COLOR_TYPE = "color"
    SEGMENTATION_TYPE = "segmentation"
