import logging
import math
from argparse import Namespace
from enum import Enum
from shutil import rmtree
from os.path import join
from os import makedirs
from abc import ABC, abstractmethod
from shutil import rmtree
from os.path import join
from os import makedirs
from typing import (
    Tuple,
    Type,
    Union,
    Dict,
    Any,
    TYPE_CHECKING,
    TypeVar,
    Generic,
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
    from wkcuber.api.Dataset import AbstractDataset, TiffDataset
from wkcuber.api.MagDataset import (
    MagDataset,
    WKMagDataset,
    TiffMagDataset,
    TiledTiffMagDataset,
    find_mag_path_on_disk,
    GenericTiffMagDataset,
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


MagT = TypeVar("MagT", bound=MagDataset)


class Layer(Generic[MagT]):

    COLOR_TYPE = "color"
    SEGMENTATION_TYPE = "segmentation"

    def __init__(
        self,
        name: str,
        dataset: "AbstractDataset",
        dtype_per_channel: np.dtype,
        num_channels: int,
    ) -> None:
        self.name = name
        self.dataset = dataset
        self.dtype_per_channel = dtype_per_channel
        self.num_channels = num_channels
        self.mags: Dict[str, MagT] = {}

        full_path = join(dataset.path, name)
        makedirs(full_path, exist_ok=True)

    def get_mag(self, mag: Union[int, str, list, tuple, np.ndarray, Mag]) -> MagT:
        mag = Mag(mag).to_layer_name()
        if mag not in self.mags.keys():
            raise IndexError("The mag {} is not a mag of this layer".format(mag))
        return self.mags[mag]

    def add_mag(self, mag: Union[int, str, list, tuple, np.ndarray, Mag]) -> MagT:
        pass

    def get_or_add_mag(
        self, mag: Union[int, str, list, tuple, np.ndarray, Mag]
    ) -> MagT:
        pass

    def delete_mag(self, mag: Union[int, str, list, tuple, np.ndarray, Mag]) -> None:
        mag = Mag(mag).to_layer_name()
        if mag not in self.mags.keys():
            raise IndexError(
                "Deleting mag {} failed. There is no mag with this name".format(mag)
            )

        del self.mags[mag]
        self.dataset.properties._delete_mag(self.name, mag)
        # delete files on disk
        full_path = find_mag_path_on_disk(self.dataset.path, self.name, mag)
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
        self.dataset.properties._set_bounding_box_of_layer(self.name, offset, size)
        bounding_box = BoundingBox(offset, size)

        for mag_name, mag in self.mags.items():
            mag.view.size = cast(
                Tuple[int, int, int],
                tuple(
                    bounding_box.align_with_mag(Mag(mag_name), ceil=True)
                    .in_mag(Mag(mag_name))
                    .bottomright
                ),
            )

    def set_bounding_box_offset(self, offset: Tuple[int, int, int]) -> None:
        size: Tuple[int, int, int] = self.dataset.properties.get_bounding_box_of_layer(
            self.name
        )[1]
        self.set_bounding_box(offset, size)

    def set_bounding_box_size(self, size: Tuple[int, int, int]) -> None:
        offset: Tuple[
            int, int, int
        ] = self.dataset.properties.get_bounding_box_of_layer(self.name)[0]
        self.set_bounding_box(offset, size)

    def _initialize_mag_from_other_mag(
        self, new_mag_name: Union[str, Mag], other_mag: MagDataset, compress: bool
    ) -> MagDataset:
        raise NotImplemented

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
        assert (
            from_mag.to_layer_name() in self.mags.keys()
        ), f"Failed to downsample data. The from_mag ({from_mag}) does not exist."

        parsed_interpolation_mode = parse_interpolation_mode(
            interpolation_mode, self.dataset.properties.data_layers[self.name].category
        )

        assert from_mag <= target_mag
        assert target_mag.to_layer_name() not in self.mags

        prev_mag_ds = self.mags[from_mag.to_layer_name()]

        mag_factors = [
            t // s for (t, s) in zip(target_mag.to_array(), from_mag.to_array())
        ]

        # initialize the new mag
        target_mag_ds = self._initialize_mag_from_other_mag(
            target_mag, prev_mag_ds, compress
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
        target_mag_view = target_mag_ds.get_view(
            offset=aligned_target_bb.topleft,
            size=aligned_target_bb.size,
            is_bounded=not compress,
        )

        # Source view
        # Setting "is_bounded" first to "False" and the to "True" temporarily disables the "bounds check".
        # This is not ideal, but we know what we are doing.
        # The reason why there might be an error otherwise is that the view is aligned with the target_mag
        # (not just with the from_mag). In this case we want that the view is aligned to the target_mag
        # because this makes it very easy to downsample data from the source to the target.
        source_view = prev_mag_ds.get_view(
            offset=aligned_source_bb.topleft,
            size=aligned_source_bb.size,
            is_bounded=False,
        )
        source_view.is_bounded = True

        # perform downsampling
        with get_executor_for_args(args) as executor:
            voxel_count_per_cube = np.prod(prev_mag_ds._get_file_dimensions())
            job_count_per_log = math.ceil(
                1024 ** 3 / voxel_count_per_cube
            )  # log every gigavoxel of processed data

            if buffer_edge_len is None:
                buffer_edge_len = determine_buffer_edge_len(
                    prev_mag_ds.view
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
                target_view=target_mag_view,
                source_chunk_size=np.array(target_mag_ds._get_file_dimensions())
                * mag_factors,
                target_chunk_size=target_mag_ds._get_file_dimensions(),
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

    def setup_mag(self, mag: Union[str, Mag]) -> None:
        pass

    def upsample(
        self,
        from_mag: Mag,
        min_mag: Optional[Mag],
        compress: bool,
        sampling_mode: str = SamplingModes.AUTO,
        buffer_edge_len: Optional[int] = None,
        args: Optional[Namespace] = None,
    ) -> None:
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

            prev_mag_ds = self.mags[prev_mag.to_layer_name()]

            mag_factors = [
                t / s for (t, s) in zip(target_mag.to_array(), prev_mag.to_array())
            ]

            # initialize the new mag
            target_mag_ds = self._initialize_mag_from_other_mag(
                target_mag, prev_mag_ds, compress
            )

            # Get target view
            target_mag_view = target_mag_ds.get_view(is_bounded=not compress)

            # perform upsampling
            with get_executor_for_args(args) as executor:
                voxel_count_per_cube = np.prod(prev_mag_ds._get_file_dimensions())
                job_count_per_log = math.ceil(
                    1024 ** 3 / voxel_count_per_cube
                )  # log every gigavoxel of processed data

                if buffer_edge_len is None:
                    buffer_edge_len = determine_buffer_edge_len(
                        prev_mag_ds.view
                    )  # DEFAULT_EDGE_LEN
                func = named_partial(
                    upsample_cube_job,
                    mag_factors=mag_factors,
                    buffer_edge_len=buffer_edge_len,
                    compress=compress,
                    job_count_per_log=job_count_per_log,
                )
                prev_mag_ds.get_view().for_zipped_chunks(
                    # this view is restricted to the bounding box specified in the properties
                    func,
                    target_view=target_mag_view,
                    source_chunk_size=target_mag_ds._get_file_dimensions(),
                    target_chunk_size=target_mag_ds._get_file_dimensions()
                    * np.array([int(1 / f) for f in mag_factors]),
                    executor=executor,
                )

            logging.info("Mag {0} successfully cubed".format(target_mag))

            prev_mag = target_mag
            target_mag = get_previous_mag(target_mag, scale)


class WKLayer(Layer[WKMagDataset]):
    mags: Dict[str, WKMagDataset]

    def add_mag(
        self,
        mag: Union[int, str, list, tuple, np.ndarray, Mag],
        block_len: int = 32,
        file_len: int = DEFAULT_WKW_FILE_LEN,
        block_type: int = wkw.Header.BLOCK_TYPE_RAW,
    ) -> WKMagDataset:
        # normalize the name of the mag
        mag = Mag(mag).to_layer_name()

        self._assert_mag_does_not_exist_yet(mag)
        self._create_dir_for_mag(mag)

        self.mags[mag] = WKMagDataset(
            self, mag, block_len, file_len, block_type, create=True
        )
        self.dataset.properties._add_mag(
            self.name, mag, cube_length=block_len * file_len
        )

        return self.mags[mag]

    def get_or_add_mag(
        self,
        mag: Union[int, str, list, tuple, np.ndarray, Mag],
        block_len: int = 32,
        file_len: int = DEFAULT_WKW_FILE_LEN,
        block_type: int = wkw.Header.BLOCK_TYPE_RAW,
    ) -> WKMagDataset:
        # normalize the name of the mag
        mag = Mag(mag).to_layer_name()

        if mag in self.mags.keys():
            assert (
                block_len is None or self.mags[mag].header.block_len == block_len
            ), f"Cannot get_or_add_mag: The mag {mag} already exists, but the block lengths do not match"
            assert (
                file_len is None or self.mags[mag].header.file_len == file_len
            ), f"Cannot get_or_add_mag: The mag {mag} already exists, but the file lengths do not match"
            assert (
                block_type is None or self.mags[mag].header.block_type == block_type
            ), f"Cannot get_or_add_mag: The mag {mag} already exists, but the block types do not match"
            return self.get_mag(mag)
        else:
            return self.add_mag(
                mag, block_len=block_len, file_len=file_len, block_type=block_type
            )

    def setup_mag(self, mag: Union[str, Mag]) -> None:
        # This method is used to initialize the mag when opening the Dataset. This does not create e.g. the wk_header.

        # normalize the name of the mag
        mag = Mag(mag).to_layer_name()

        self._assert_mag_does_not_exist_yet(mag)

        try:
            with wkw.Dataset.open(
                str(find_mag_path_on_disk(self.dataset.path, self.name, mag))
            ) as wkw_dataset:
                wk_header = wkw_dataset.header

            self.mags[mag] = WKMagDataset(
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
        self, new_mag_name: Union[str, Mag], other_mag: MagDataset, compress: bool
    ) -> MagDataset:
        block_type = (
            wkw.Header.BLOCK_TYPE_LZ4HC if compress else wkw.Header.BLOCK_TYPE_RAW
        )
        other_wk_mag = cast(
            WKMagDataset, other_mag
        )  # This method is only used in the context of creating a new magnification by using the same meta data as another magnification of the same dataset
        return self.add_mag(
            new_mag_name,
            block_len=other_wk_mag.block_len,
            file_len=other_wk_mag.file_len,
            block_type=block_type,
        )


TiffMagT = TypeVar("TiffMagT", bound=GenericTiffMagDataset)


class GenericTiffLayer(Layer[TiffMagT], ABC):
    dataset: "TiffDataset"

    def add_mag(self, mag: Union[int, str, list, tuple, np.ndarray, Mag]) -> TiffMagT:
        # normalize the name of the mag
        mag = Mag(mag).to_layer_name()

        self._assert_mag_does_not_exist_yet(mag)
        self._create_dir_for_mag(mag)

        self.mags[mag] = self._get_mag_dataset_class()(
            self, mag, self.dataset.properties.pattern
        )
        self.dataset.properties._add_mag(self.name, mag)

        return self.mags[mag]

    def get_or_add_mag(
        self, mag: Union[int, str, list, tuple, np.ndarray, Mag]
    ) -> TiffMagT:
        # normalize the name of the mag
        mag = Mag(mag).to_layer_name()

        if mag in self.mags.keys():
            return self.get_mag(mag)
        else:
            return self.add_mag(mag)

    def setup_mag(self, mag: Union[str, Mag]) -> None:
        # This method is used to initialize the mag when opening the Dataset. This does not create e.g. folders.

        # normalize the name of the mag
        mag = Mag(mag).to_layer_name()

        self._assert_mag_does_not_exist_yet(mag)

        self.mags[mag] = self._get_mag_dataset_class()(
            self, mag, self.dataset.properties.pattern
        )
        self.dataset.properties._add_mag(self.name, mag)

    @abstractmethod
    def _get_mag_dataset_class(self) -> Type[TiffMagT]:
        pass


class TiffLayer(GenericTiffLayer[TiffMagDataset]):
    def _get_mag_dataset_class(self) -> Type[TiffMagDataset]:
        return TiffMagDataset

    def _initialize_mag_from_other_mag(
        self, new_mag_name: Union[str, Mag], other_mag: MagDataset, compress: bool
    ) -> MagDataset:
        return self.add_mag(new_mag_name)


class TiledTiffLayer(GenericTiffLayer[TiledTiffMagDataset]):
    def _get_mag_dataset_class(self) -> Type[TiledTiffMagDataset]:
        return TiledTiffMagDataset

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
        raise NotImplemented

    def downsample_mag(
        self,
        from_mag: Mag,
        target_mag: Mag,
        interpolation_mode: str = "default",
        compress: bool = True,
        buffer_edge_len: Optional[int] = None,
        args: Optional[Namespace] = None,
    ) -> None:
        raise NotImplemented

    def downsample_mag_list(
        self,
        from_mag: Mag,
        target_mags: List[Mag],
        interpolation_mode: str = "default",
        compress: bool = True,
        buffer_edge_len: Optional[int] = None,
        args: Optional[Namespace] = None,
    ) -> None:
        raise NotImplemented
