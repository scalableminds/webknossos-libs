import os
from os.path import join
from pathlib import Path
from typing import Type, Tuple, Union, cast, TYPE_CHECKING, TypeVar, Generic, Any

from wkw import wkw
import numpy as np

import wkcuber.api as api

if TYPE_CHECKING:
    from wkcuber.api.Layer import (
        WKLayer,
        Layer,
        GenericTiffLayer,
        TiffLayer,
        TiledTiffLayer,
    )
from wkcuber.api.View import WKView, TiffView, View
from wkcuber.api.TiffData.TiffMag import TiffMagHeader
from wkcuber.mag import Mag


def find_mag_path_on_disk(
    dataset_path: Union[str, Path], layer_name: str, mag_name: str
) -> str:
    mag = Mag(mag_name)
    short_mag_file_path = join(dataset_path, layer_name, mag.to_layer_name())
    long_mag_file_path = join(dataset_path, layer_name, mag.to_long_layer_name())
    if os.path.exists(short_mag_file_path):
        return short_mag_file_path
    else:
        return long_mag_file_path


class MagDataset:
    def __init__(self, layer: "Layer", name: str) -> None:
        self.layer = layer
        self.name = name
        self.header = self.get_header()

        self.view = self.get_view(offset=(0, 0, 0), is_bounded=False)

    def open(self) -> None:
        self.view.open()

    def close(self) -> None:
        self.view.close()

    def read(
        self,
        offset: Tuple[int, int, int] = (0, 0, 0),
        size: Tuple[int, int, int] = None,
    ) -> np.array:
        return self.view.read(offset, size)

    def write(
        self,
        data: np.ndarray,
        offset: Tuple[int, int, int] = (0, 0, 0),
        allow_compressed_write: bool = False,
    ) -> None:
        self._assert_valid_num_channels(data.shape)
        self.view.write(data, offset, allow_compressed_write)
        layer_properties = self.layer.dataset.properties.data_layers[self.layer.name]
        current_offset_in_mag1 = layer_properties.get_bounding_box_offset()
        current_size_in_mag1 = layer_properties.get_bounding_box_size()

        mag = Mag(self.name)
        mag_np = mag.as_np()

        offset_in_mag1 = tuple(np.array(offset) * mag_np)

        new_offset_in_mag1 = (
            offset_in_mag1
            if current_offset_in_mag1 == (-1, -1, -1)
            else tuple(min(x) for x in zip(current_offset_in_mag1, offset_in_mag1))
        )

        old_end_offset_in_mag1 = np.array(current_offset_in_mag1) + np.array(
            current_size_in_mag1
        )
        new_end_offset_in_mag1 = (np.array(offset) + np.array(data.shape[-3:])) * mag_np
        max_end_offset_in_mag1 = np.array(
            [old_end_offset_in_mag1, new_end_offset_in_mag1]
        ).max(axis=0)
        total_size_in_mag1 = max_end_offset_in_mag1 - np.array(new_offset_in_mag1)
        total_size = total_size_in_mag1 / mag_np

        self.view.size = cast(Tuple[int, int, int], tuple(total_size))

        self.layer.dataset.properties._set_bounding_box_of_layer(
            self.layer.name,
            cast(Tuple[int, int, int], tuple(new_offset_in_mag1)),
            cast(Tuple[int, int, int], tuple(total_size_in_mag1)),
        )

    def get_header(self) -> Union[TiffMagHeader, wkw.Header]:
        raise NotImplementedError

    def get_view(
        self,
        size: Tuple[int, int, int] = None,
        offset: Tuple[int, int, int] = None,
        is_bounded: bool = True,
    ) -> View:
        size_in_properties = self.layer.dataset.properties.data_layers[
            self.layer.name
        ].get_bounding_box_size()

        offset_in_properties = self.layer.dataset.properties.data_layers[
            self.layer.name
        ].get_bounding_box_offset()

        if offset_in_properties == (-1, -1, -1):
            offset_in_properties = (0, 0, 0)

        if offset is None:
            offset = offset_in_properties

        if size is None:
            size = np.array(size_in_properties) - (
                np.array(offset) - np.array(offset_in_properties)
            )

        # assert that the parameters size and offset are valid
        if is_bounded:
            for off_prop, off in zip(offset_in_properties, offset):
                if off < off_prop:
                    raise AssertionError(
                        f"The passed parameter 'offset' {offset} is outside the bounding box from the properties.json. "
                        f"Use is_bounded=False if you intend to write outside out the existing bounding box."
                    )
            for s1, s2, off in zip(size_in_properties, size, offset):
                if s2 + off > s1:
                    raise AssertionError(
                        f"The combination of the passed parameter 'size' {size} and 'offset' {offset} are not compatible with the "
                        f"size ({size_in_properties}) from the properties.json.  "
                        f"Use is_bounded=False if you intend to write outside out the existing bounding box."
                    )

        mag_file_path = find_mag_path_on_disk(
            self.layer.dataset.path, self.layer.name, self.name
        )
        return self._get_view_type()(
            mag_file_path, self.header, size, offset, is_bounded
        )

    def _get_view_type(self) -> Type[View]:
        raise NotImplementedError

    def _assert_valid_num_channels(self, write_data_shape: Tuple[int, ...]) -> None:
        num_channels = self.layer.num_channels
        if len(write_data_shape) == 3:
            assert (
                num_channels == 1
            ), f"The number of channels of the dataset ({num_channels}) does not match the number of channels of the passed data (1)"
        else:
            assert (
                num_channels == write_data_shape[0]
            ), f"The number of channels of the dataset ({num_channels}) does not match the number of channels of the passed data ({write_data_shape[0]})"


class WKMagDataset(MagDataset):
    header: wkw.Header

    def __init__(
        self,
        layer: "WKLayer",
        name: str,
        block_len: int,
        file_len: int,
        block_type: int,
        create: bool = False,
    ) -> None:
        self.block_len = block_len
        self.file_len = file_len
        self.block_type = block_type
        super().__init__(layer, name)
        if create:
            wkw.Dataset.create(
                join(layer.dataset.path, layer.name, self.name), self.header
            )

    def get_header(self) -> wkw.Header:
        return wkw.Header(
            voxel_type=self.layer.dtype_per_channel,
            num_channels=self.layer.num_channels,
            version=1,
            block_len=self.block_len,
            file_len=self.file_len,
            block_type=self.block_type,
        )

    def _get_view_type(self) -> Type[WKView]:
        return WKView


TiffLayerT = TypeVar("TiffLayerT", bound="GenericTiffLayer")


class GenericTiffMagDataset(MagDataset, Generic[TiffLayerT]):
    layer: TiffLayerT

    def __init__(self, layer: TiffLayerT, name: str, pattern: str) -> None:
        self.pattern = pattern
        super().__init__(layer, name)

    def get_header(self) -> TiffMagHeader:
        return TiffMagHeader(
            pattern=self.pattern,
            dtype_per_channel=self.layer.dtype_per_channel,
            num_channels=self.layer.num_channels,
            tile_size=self.layer.dataset.properties.tile_size,
        )

    def _get_view_type(self) -> Type[TiffView]:
        return TiffView


class TiffMagDataset(GenericTiffMagDataset["TiffLayer"]):
    pass


class TiledTiffMagDataset(GenericTiffMagDataset["TiledTiffLayer"]):
    def get_tile(self, x_index: int, y_index: int, z_index: int) -> np.array:
        tile_size = self.layer.dataset.properties.tile_size
        assert tile_size is not None
        size = (tile_size[0], tile_size[1], 1)
        offset = np.array((0, 0, 0)) + np.array(size) * np.array(
            (x_index, y_index, z_index)
        )
        return self.read(offset, size)
