import operator
from argparse import Namespace
from shutil import rmtree
from abc import ABC, abstractmethod
from os import makedirs, path
from os.path import join, normpath, basename
from pathlib import Path
from typing import Type, Tuple, Union, Dict, Any, Optional, cast, TypeVar, Generic

import numpy as np
import os
import re

from wkcuber.api.bounding_box import BoundingBox
from wkcuber.mag import Mag
from wkcuber.utils import logger, get_executor_for_args, ceil_div_np

from wkcuber.api.Properties.DatasetProperties import (
    WKProperties,
    TiffProperties,
    Properties,
)
from wkcuber.api.Layer import Layer, WKLayer, TiffLayer, TiledTiffLayer
from wkcuber.api.View import View

DEFAULT_BIT_DEPTH = 8


def is_int(s: str) -> bool:
    try:
        int(s)
        return True
    except ValueError:
        return False


def convert_dtypes(
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
        (str(int(op(int(part), num_channels))) if is_int(part) else part)
        for part in dtype_parts
    ]
    return "".join(converted_dtype_parts)


def dtype_per_layer_to_dtype_per_channel(
    dtype_per_layer: Union[str, np.dtype], num_channels: int
) -> np.dtype:
    try:
        return np.dtype(
            convert_dtypes(
                dtype_per_layer, num_channels, dtype_per_layer_to_dtype_per_channel=True
            )
        )
    except TypeError as e:
        raise TypeError(
            "Converting dtype_per_layer to dtype_per_channel failed. Double check if the dtype_per_layer value is correct. "
            + str(e)
        )


def dtype_per_channel_to_dtype_per_layer(
    dtype_per_channel: Union[str, np.dtype], num_channels: int
) -> str:
    return convert_dtypes(
        np.dtype(dtype_per_channel),
        num_channels,
        dtype_per_layer_to_dtype_per_channel=False,
    )


def copy_job(args: Tuple[View, View, int]) -> None:
    (source_view, target_view, i) = args
    # Copy the data form one view to the other in a buffered fashion
    target_view.write(source_view.read())


LayerT = TypeVar("LayerT", bound=Layer)


class AbstractDataset(Generic[LayerT]):
    @abstractmethod
    def __init__(self, dataset_path: Union[str, Path]) -> None:
        properties: Properties = self._get_properties_type()._from_json(
            join(dataset_path, Properties.FILE_NAME)
        )
        self.layers: Dict[str, LayerT] = {}
        self.path = Path(properties.path).parent
        self.properties = properties
        self._data_format = "abstract"

        # construct self.layer
        for layer_name in self.properties.data_layers:
            layer = self.properties.data_layers[layer_name]
            self.add_layer(
                layer.name,
                layer.category,
                dtype_per_layer=layer.element_class,
                num_channels=layer.num_channels,
            )
            for resolution in layer.wkw_magnifications:
                self.layers[layer_name].setup_mag(resolution.mag.to_layer_name())

    @classmethod
    def create_with_properties(cls, properties: Properties) -> "AbstractDataset":
        dataset_path = path.dirname(properties.path)

        if os.path.exists(dataset_path):
            assert os.path.isdir(
                dataset_path
            ), f"Creation of Dataset at {dataset_path} failed, because a file already exists at this path."
            assert not os.listdir(
                dataset_path
            ), f"Creation of Dataset at {dataset_path} failed, because a non-empty folder already exists at this path."

        # create directories on disk and write datasource-properties.json
        try:
            makedirs(dataset_path, exist_ok=True)
            properties._export_as_json()
        except OSError as e:
            raise type(e)(
                "Creation of Dataset {} failed. ".format(dataset_path) + repr(e)
            )

        # initialize object
        return cls(dataset_path)

    def get_properties(self) -> Properties:
        return self.properties

    def get_layer(self, layer_name: str) -> LayerT:
        if layer_name not in self.layers.keys():
            raise IndexError(
                "The layer {} is not a layer of this dataset".format(layer_name)
            )
        return self.layers[layer_name]

    def add_layer(
        self,
        layer_name: str,
        category: str,
        dtype_per_layer: Union[str, np.dtype] = None,
        dtype_per_channel: Union[str, np.dtype] = None,
        num_channels: int = None,
        **kwargs: Any,
    ) -> LayerT:
        if "dtype" in kwargs:
            raise ValueError(
                f"Called Dataset.add_layer with 'dtype'={kwargs['dtype']}. This parameter is deprecated. Use 'dtype_per_layer' or 'dtype_per_channel' instead."
            )
        if num_channels is None:
            num_channels = 1

        if dtype_per_layer is not None and dtype_per_channel is not None:
            raise AttributeError(
                "Cannot add layer. Specifying both 'dtype_per_layer' and 'dtype_per_channel' is not allowed"
            )
        elif dtype_per_channel is not None:
            try:
                dtype_per_channel = np.dtype(dtype_per_channel)
            except TypeError as e:
                raise TypeError(
                    "Cannot add layer. The specified 'dtype_per_channel' must be a valid dtype. "
                    + str(e)
                )
            dtype_per_layer = dtype_per_channel_to_dtype_per_layer(
                dtype_per_channel, num_channels
            )
        elif dtype_per_layer is not None:
            try:
                dtype_per_layer = str(np.dtype(dtype_per_layer))
            except Exception:
                pass  # casting to np.dtype fails if the user specifies a special dtype like "uint24"
            dtype_per_channel = dtype_per_layer_to_dtype_per_channel(
                dtype_per_layer, num_channels
            )
        else:
            dtype_per_layer = "uint" + str(DEFAULT_BIT_DEPTH * num_channels)
            dtype_per_channel = np.dtype("uint" + str(DEFAULT_BIT_DEPTH))

        if layer_name in self.layers.keys():
            raise IndexError(
                "Adding layer {} failed. There is already a layer with this name".format(
                    layer_name
                )
            )

        self.properties._add_layer(
            layer_name,
            category,
            dtype_per_layer,
            self._data_format,
            num_channels,
            **kwargs,
        )
        self.layers[layer_name] = self._create_layer(
            layer_name, dtype_per_channel, num_channels
        )
        return self.layers[layer_name]

    def get_or_add_layer(
        self,
        layer_name: str,
        category: str,
        dtype_per_layer: Union[str, np.dtype] = None,
        dtype_per_channel: Union[str, np.dtype] = None,
        num_channels: int = None,
        **kwargs: Any,
    ) -> LayerT:
        if "dtype" in kwargs:
            raise ValueError(
                f"Called Dataset.get_or_add_layer with 'dtype'={kwargs['dtype']}. This parameter is deprecated. Use 'dtype_per_layer' or 'dtype_per_channel' instead."
            )
        if layer_name in self.layers.keys():
            assert (
                num_channels is None
                or self.layers[layer_name].num_channels == num_channels
            ), (
                f"Cannot get_or_add_layer: The layer '{layer_name}' already exists, but the number of channels do not match. "
                + f"The number of channels of the existing layer are '{self.layers[layer_name].num_channels}' "
                + f"and the passed parameter is '{num_channels}'."
            )
            assert self.properties.data_layers[layer_name].category == category, (
                f"Cannot get_or_add_layer: The layer '{layer_name}' already exists, but the categories do not match. "
                + f"The category of the existing layer is '{self.properties.data_layers[layer_name].category}' "
                + f"and the passed parameter is '{category}'."
            )

            if dtype_per_channel is not None or dtype_per_layer is not None:
                dtype_per_channel = (
                    dtype_per_channel
                    or dtype_per_layer_to_dtype_per_channel(
                        dtype_per_layer,
                        num_channels or self.layers[layer_name].num_channels,
                    )
                )
                assert (
                    dtype_per_channel is None
                    or self.layers[layer_name].dtype_per_channel == dtype_per_channel
                ), (
                    f"Cannot get_or_add_layer: The layer '{layer_name}' already exists, but the dtypes do not match. "
                    + f"The dtype_per_channel of the existing layer is '{self.layers[layer_name].dtype_per_channel}' "
                    + f"and the passed parameter would result in a dtype_per_channel of '{dtype_per_channel}'."
                )
            return self.layers[layer_name]
        else:
            return self.add_layer(
                layer_name,
                category,
                dtype_per_layer=dtype_per_layer,
                dtype_per_channel=dtype_per_channel,
                num_channels=num_channels,
                **kwargs,
            )

    def delete_layer(self, layer_name: str) -> None:
        if layer_name not in self.layers.keys():
            raise IndexError(
                f"Removing layer {layer_name} failed. There is no layer with this name"
            )
        del self.layers[layer_name]
        self.properties._delete_layer(layer_name)
        # delete files on disk
        rmtree(join(self.path, layer_name))

    def add_symlink_layer(self, foreign_layer_path: Union[str, Path]) -> LayerT:
        foreign_layer_path = os.path.abspath(foreign_layer_path)
        layer_name = os.path.basename(os.path.normpath(foreign_layer_path))
        if layer_name in self.layers.keys():
            raise IndexError(
                f"Cannot create symlink to {foreign_layer_path}. This dataset already has a layer called {layer_name}."
            )

        os.symlink(foreign_layer_path, join(self.path, layer_name))

        # copy the properties of the layer into the properties of this dataset
        layer_properties = self._get_type()(
            Path(foreign_layer_path).parent
        ).properties.data_layers[layer_name]
        self.properties.data_layers[layer_name] = layer_properties
        self.properties._export_as_json()

        self.layers[layer_name] = self._create_layer(
            layer_name,
            dtype_per_layer_to_dtype_per_channel(
                layer_properties.element_class, layer_properties.num_channels
            ),
            layer_properties.num_channels,
        )
        for resolution in layer_properties.wkw_magnifications:
            self.layers[layer_name].setup_mag(resolution.mag.to_layer_name())
        return self.layers[layer_name]

    def get_view(
        self,
        layer_name: str,
        mag: Union[int, str, list, tuple, np.ndarray, Mag],
        size: Tuple[int, int, int],
        offset: Tuple[int, int, int] = None,
        is_bounded: bool = True,
        read_only: bool = False,
    ) -> View:
        layer = self.get_layer(layer_name)
        mag_ds = layer.get_mag(mag)

        return mag_ds.get_view(
            size=size, offset=offset, is_bounded=is_bounded, read_only=read_only
        )

    def _create_layer(
        self, layer_name: str, dtype_per_channel: np.dtype, num_channels: int
    ) -> LayerT:
        raise NotImplementedError

    def copy_dataset(
        self, empty_target_ds: "AbstractDataset", args: Optional[Namespace] = None
    ) -> None:
        assert (
            len(empty_target_ds.layers) == 0
        ), f"Copying dataset failed. The target dataset must be empty."
        with get_executor_for_args(args) as executor:
            for layer_name, layer in self.layers.items():
                largest_segment_id = None
                if (
                    self.properties.data_layers[layer_name].category
                    == Layer.SEGMENTATION_TYPE
                ):
                    largest_segment_id = self.properties.data_layers[
                        layer_name
                    ].largest_segment_id
                target_layer = empty_target_ds.add_layer(
                    layer_name,
                    self.properties.data_layers[layer_name].category,
                    dtype_per_channel=layer.dtype_per_channel,
                    num_channels=layer.num_channels,
                    largest_segment_id=largest_segment_id,
                )

                bbox = self.properties.get_bounding_box_of_layer(layer_name)

                for mag_name, mag in layer.mags.items():
                    target_mag = target_layer.add_mag(mag_name)

                    # The bounding box needs to be updated manually because chunked views do not have a reference to the dataset itself
                    # The base view of a MagDataset always starts at (0, 0, 0)
                    target_mag.view.global_offset = (0, 0, 0)
                    target_mag.view.size = cast(
                        Tuple[int, int, int],
                        tuple(
                            BoundingBox(topleft=bbox[0], size=bbox[1])
                            .align_with_mag(Mag(mag_name), ceil=True)
                            .in_mag(Mag(mag_name))
                            .bottomright
                        ),
                    )
                    target_mag.layer.dataset.properties._set_bounding_box_of_layer(
                        layer_name, offset=bbox[0], size=bbox[1]
                    )

                    # The data gets written to the target_mag.
                    # Therefore, the chunk size is determined by the target_mag to prevent concurrent writes
                    mag.view.for_zipped_chunks(
                        work_on_chunk=copy_job,
                        target_view=target_mag.view,
                        source_chunk_size=target_mag._get_file_dimensions(),
                        target_chunk_size=target_mag._get_file_dimensions(),
                        executor=executor,
                    )

    def to_wk_dataset(
        self,
        new_dataset_path: Union[str, Path],
        scale: Optional[Tuple[float, float, float]] = None,
    ) -> "WKDataset":
        if scale is None:
            scale = self.properties.scale
        new_ds = WKDataset.create(new_dataset_path, scale=scale)
        self.copy_dataset(new_ds)
        return new_ds

    def to_tiff_dataset(
        self,
        new_dataset_path: Union[str, Path],
        scale: Optional[Tuple[float, float, float]] = None,
        pattern: Optional[str] = None,
    ) -> "TiffDataset":
        if scale is None:
            scale = self.properties.scale
        new_ds = TiffDataset.create(new_dataset_path, scale=scale, pattern=pattern)
        self.copy_dataset(new_ds)
        return new_ds

    def to_tiled_tiff_dataset(
        self,
        new_dataset_path: Union[str, Path],
        tile_size: Tuple[int, int],
        scale: Optional[Tuple[float, float, float]] = None,
        pattern: Optional[str] = None,
    ) -> "TiledTiffDataset":
        if scale is None:
            scale = self.properties.scale
        new_ds = TiledTiffDataset.create(
            new_dataset_path, scale=scale, tile_size=tile_size, pattern=pattern
        )
        self.copy_dataset(new_ds)
        return new_ds

    @abstractmethod
    def _get_properties_type(self) -> Type[Properties]:
        pass

    @abstractmethod
    def _get_type(self) -> Type["AbstractDataset"]:
        pass


class WKDataset(AbstractDataset[WKLayer]):
    @classmethod
    def create(
        cls, dataset_path: Union[str, Path], scale: Tuple[float, float, float]
    ) -> "WKDataset":
        name = basename(normpath(dataset_path))
        properties = WKProperties(join(dataset_path, Properties.FILE_NAME), name, scale)
        return cast(WKDataset, WKDataset.create_with_properties(properties))

    @classmethod
    def get_or_create(
        cls, dataset_path: Union[str, Path], scale: Tuple[float, float, float]
    ) -> "WKDataset":
        if os.path.exists(
            join(dataset_path, Properties.FILE_NAME)
        ):  # use the properties file to check if the Dataset exists
            ds = WKDataset(dataset_path)
            assert tuple(ds.properties.scale) == tuple(
                scale
            ), f"Cannot get_or_create WKDataset: The dataset {dataset_path} already exists, but the scales do not match ({ds.properties.scale} != {scale})"
            return ds
        else:
            return cls.create(dataset_path, scale)

    def __init__(self, dataset_path: Union[str, Path]) -> None:
        super().__init__(dataset_path)
        self._data_format = "wkw"
        assert isinstance(self.properties, WKProperties)

    def _create_layer(
        self, layer_name: str, dtype_per_channel: np.dtype, num_channels: int
    ) -> WKLayer:
        return WKLayer(layer_name, self, dtype_per_channel, num_channels)

    def _get_properties_type(self) -> Type[WKProperties]:
        return WKProperties

    def _get_type(self) -> Type["WKDataset"]:
        return WKDataset


class TiffDataset(AbstractDataset[TiffLayer]):
    properties: TiffProperties

    @classmethod
    def create(
        cls,
        dataset_path: Union[str, Path],
        scale: Tuple[float, float, float],
        pattern: Optional[str] = None,
    ) -> "TiffDataset":
        if pattern is None:
            pattern = "{zzzzz}.tif"
        validate_pattern(pattern)
        name = basename(normpath(dataset_path))
        properties = TiffProperties(
            join(dataset_path, "datasource-properties.json"),
            name,
            scale,
            pattern=pattern,
            tile_size=None,
        )
        return cast(TiffDataset, TiffDataset.create_with_properties(properties))

    @classmethod
    def get_or_create(
        cls,
        dataset_path: Union[str, Path],
        scale: Tuple[float, float, float],
        pattern: str = None,
    ) -> "TiffDataset":
        if os.path.exists(
            join(dataset_path, Properties.FILE_NAME)
        ):  # use the properties file to check if the Dataset exists
            ds = TiffDataset(dataset_path)
            assert tuple(ds.properties.scale) == tuple(
                scale
            ), f"Cannot get_or_create TiffDataset: The dataset {dataset_path} already exists, but the scales do not match ({ds.properties.scale} != {scale})"
            if pattern is not None:
                assert (
                    ds.properties.pattern == pattern
                ), f"Cannot get_or_create TiffDataset: The dataset {dataset_path} already exists, but the patterns do not match ({ds.properties.pattern} != {pattern})"
            return ds
        else:
            if pattern is None:
                return cls.create(dataset_path, scale)
            else:
                return cls.create(dataset_path, scale, pattern)

    def __init__(self, dataset_path: Union[str, Path]) -> None:
        super().__init__(dataset_path)
        self.data_format = "tiff"
        assert isinstance(self.properties, TiffProperties)

    def _create_layer(
        self, layer_name: str, dtype_per_channel: np.dtype, num_channels: int
    ) -> TiffLayer:
        return TiffLayer(layer_name, self, dtype_per_channel, num_channels)

    def _get_properties_type(self) -> Type[TiffProperties]:
        return TiffProperties

    def _get_type(self) -> Type["TiffDataset"]:
        return TiffDataset


class TiledTiffDataset(AbstractDataset[TiledTiffLayer]):
    properties: TiffProperties

    @classmethod
    def create(
        cls,
        dataset_path: Union[str, Path],
        scale: Tuple[float, float, float],
        tile_size: Tuple[int, int],
        pattern: Optional[str] = None,
    ) -> "TiledTiffDataset":
        if pattern is None:
            pattern = "{xxxxx}/{yyyyy}/{zzzzz}.tif"
        validate_pattern(pattern)
        name = basename(normpath(dataset_path))
        properties = TiffProperties(
            join(dataset_path, "datasource-properties.json"),
            name,
            scale,
            pattern=pattern,
            tile_size=tile_size,
        )
        return cast(
            TiledTiffDataset, TiledTiffDataset.create_with_properties(properties)
        )

    @classmethod
    def get_or_create(
        cls,
        dataset_path: Union[str, Path],
        scale: Tuple[float, float, float],
        tile_size: Tuple[int, int],
        pattern: str = None,
    ) -> "TiledTiffDataset":
        if os.path.exists(
            join(dataset_path, Properties.FILE_NAME)
        ):  # use the properties file to check if the Dataset exists
            ds = TiledTiffDataset(dataset_path)
            assert tuple(ds.properties.scale) == tuple(
                scale
            ), f"Cannot get_or_create TiledTiffDataset: The dataset {dataset_path} already exists, but the scales do not match ({ds.properties.scale} != {scale})"
            assert ds.properties.tile_size is not None
            assert tuple(ds.properties.tile_size) == tuple(
                tile_size
            ), f"Cannot get_or_create TiledTiffDataset: The dataset {dataset_path} already exists, but the tile sizes do not match ({ds.properties.tile_size} != {tile_size})"
            if pattern is not None:
                assert (
                    ds.properties.pattern == pattern
                ), f"Cannot get_or_create TiledTiffDataset: The dataset {dataset_path} already exists, but the patterns do not match ({ds.properties.pattern} != {pattern})"
            return ds
        else:
            if pattern is None:
                return cls.create(dataset_path, scale, tile_size)
            else:
                return cls.create(dataset_path, scale, tile_size, pattern)

    def __init__(self, dataset_path: Union[str, Path]) -> None:
        super().__init__(dataset_path)
        self.data_format = "tiled_tiff"
        assert isinstance(self.properties, TiffProperties)

    def _create_layer(
        self, layer_name: str, dtype_per_channel: np.dtype, num_channels: int
    ) -> TiledTiffLayer:
        return TiledTiffLayer(layer_name, self, dtype_per_channel, num_channels)

    def _get_properties_type(self) -> Type[TiffProperties]:
        return TiffProperties

    def _get_type(self) -> Type["TiledTiffDataset"]:
        return TiledTiffDataset


def validate_pattern(pattern: str) -> None:
    assert pattern.count("{") > 0 and pattern.count("}") > 0, (
        f"The provided pattern {pattern} is invalid."
        + " It needs to contain at least one '{' and one '}'."
    )
    assert pattern.count("{") == pattern.count("}"), (
        f"The provided pattern {pattern} is invalid."
        + " The number of '{' does not match the number of '}'."
    )
