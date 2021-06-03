import operator
from argparse import Namespace
from shutil import rmtree
from abc import abstractmethod
from os import makedirs
from os.path import join, normpath, basename
from pathlib import Path
from typing import Type, Tuple, Union, Dict, Any, Optional, cast, TypeVar, Generic

import numpy as np
import os
import re

from wkcuber.api.Properties.LayerProperties import (
    properties_floating_type_to_python_type,
    SegmentationLayerProperties,
)
from wkcuber.api.bounding_box import BoundingBox
from wkcuber.mag import Mag
from wkcuber.utils import get_executor_for_args

from wkcuber.api.Properties.DatasetProperties import (
    WKProperties,
    TiffProperties,
    Properties,
)
from wkcuber.api.Layer import Layer, WKLayer, TiffLayer, TiledTiffLayer
from wkcuber.api.View import View

DEFAULT_BIT_DEPTH = 8


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
    return dtype_per_layer


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


def _copy_job(args: Tuple[View, View, int]) -> None:
    (source_view, target_view, _) = args
    # Copy the data form one view to the other in a buffered fashion
    target_view.write(source_view.read())


LayerT = TypeVar("LayerT", bound=Layer)


class AbstractDataset(Generic[LayerT]):
    """
    A dataset is the entry point of the Dataset API. An existing dataset on disk can be opened
    or new datasets can be created.
    """

    @abstractmethod
    def __init__(self, dataset_path: Union[str, Path]) -> None:
        """
        To open an existing dataset on disk, simply call the constructor of the appropriate dataset type (e.g. `WKDataset`).
        This requires that the `datasource-properties.json` exists. Based on the `datasource-properties.json`,
        a dataset object is constructed. Only layers and magnifications that are listed in the properties are loaded
        (even though there might exists more layer or magnifications on disk).

        The `dataset_path` refers to the top level directory of the dataset (excluding layer or magnification names).
        """

        self.path = Path(dataset_path)
        """Location of the dataset"""

        self.properties: Properties = self._get_properties_type()._from_json(
            self.path / Properties.FILE_NAME
        )
        """
        The metadata from the `datasource-properties.json`. 
        The properties are exported to disk automatically, every time the metadata changes.
        """

        self._layers: Dict[str, LayerT] = {}
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
                self.get_layer(layer_name)._setup_mag(resolution.mag.to_layer_name())

    @property
    def layers(self) -> Dict[str, LayerT]:
        """
        Getter for dictionary containing all layers.
        """
        return self._layers

    @classmethod
    def _create_with_properties(cls, properties: Properties) -> "AbstractDataset":
        dataset_dir = properties.path.parent
        if dataset_dir.exists():
            assert (
                dataset_dir.is_dir()
            ), f"Creation of Dataset at {dataset_dir} failed, because a file already exists at this path."
            assert not os.listdir(
                dataset_dir
            ), f"Creation of Dataset at {dataset_dir} failed, because a non-empty folder already exists at this path."

        # create directories on disk and write datasource-properties.json
        try:
            makedirs(dataset_dir, exist_ok=True)
            properties._export_as_json()
        except OSError as e:
            raise type(e)(
                "Creation of Dataset {} failed. ".format(dataset_dir) + repr(e)
            )

        # initialize object
        return cls(dataset_dir)

    def get_layer(self, layer_name: str) -> LayerT:
        """
        Returns the layer called `layer_name` of this dataset. The return type is `wkcuber.api.Layer.Layer`.

        This function raises an `IndexError` if the specified `layer_name` does not exist.
        """
        if layer_name not in self.layers.keys():
            raise IndexError(
                "The layer {} is not a layer of this dataset".format(layer_name)
            )
        return self.layers[layer_name]

    def add_layer(
        self,
        layer_name: str,
        category: str,
        dtype_per_layer: Union[str, np.dtype, type] = None,
        dtype_per_channel: Union[str, np.dtype, type] = None,
        num_channels: int = None,
        **kwargs: Any,
    ) -> LayerT:
        """
        Creates a new layer called `layer_name` and adds it to the dataset.
        The dtype can either be specified per layer or per channel.
        If neither of them are specified, `uint8` per channel is used as default.
        When creating a `wkcuber.api.Layer.SegmentationLayer` (category="segmentation"),
        the parameter `largest_segment_id` also has to be specified.

        The return type is `wkcuber.api.Layer.Layer`.

        This function raises an `IndexError` if the specified `layer_name` already exists.
        """
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
            dtype_per_channel = properties_floating_type_to_python_type.get(
                dtype_per_channel, dtype_per_channel
            )
            dtype_per_channel = _normalize_dtype_per_channel(dtype_per_channel)
            dtype_per_layer = _dtype_per_channel_to_dtype_per_layer(
                dtype_per_channel, num_channels
            )
        elif dtype_per_layer is not None:
            dtype_per_layer = properties_floating_type_to_python_type.get(
                dtype_per_layer, dtype_per_layer
            )
            dtype_per_layer = _normalize_dtype_per_layer(dtype_per_layer)
            dtype_per_channel = _dtype_per_layer_to_dtype_per_channel(
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
        self._layers[layer_name] = self._create_layer(
            layer_name, dtype_per_channel, num_channels
        )
        return self.layers[layer_name]

    def get_or_add_layer(
        self,
        layer_name: str,
        category: str,
        dtype_per_layer: Union[str, np.dtype, type] = None,
        dtype_per_channel: Union[str, np.dtype, type] = None,
        num_channels: int = None,
        **kwargs: Any,
    ) -> LayerT:
        """
        Creates a new layer called `layer_name` and adds it to the dataset, in case it did not exist before.
        Then, returns the layer.

        For more information see `add_layer`.
        """

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

            if dtype_per_channel is not None:
                dtype_per_channel = _normalize_dtype_per_channel(dtype_per_channel)

            if dtype_per_layer is not None:
                dtype_per_layer = _normalize_dtype_per_layer(dtype_per_layer)

            if dtype_per_channel is not None or dtype_per_layer is not None:
                dtype_per_channel = (
                    dtype_per_channel
                    or _dtype_per_layer_to_dtype_per_channel(
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
        """
        Deletes the layer from the `datasource-properties.json` and the data from disk.
        """

        if layer_name not in self.layers.keys():
            raise IndexError(
                f"Removing layer {layer_name} failed. There is no layer with this name"
            )
        del self._layers[layer_name]
        self.properties._delete_layer(layer_name)
        # delete files on disk
        rmtree(join(self.path, layer_name))

    def add_symlink_layer(self, foreign_layer_path: Union[str, Path]) -> LayerT:
        """
        Creates a symlink to the data at `foreign_layer_path` which belongs to another dataset.
        The relevant information from the `datasource-properties.json` of the other dataset is copied to this dataset.
        Note: If the other dataset modifies its bounding box afterwards, the change does not affect this properties
        (or vice versa).
        """
        foreign_layer_path = Path(os.path.abspath(foreign_layer_path))
        layer_name = foreign_layer_path.name
        if layer_name in self.layers.keys():
            raise IndexError(
                f"Cannot create symlink to {foreign_layer_path}. This dataset already has a layer called {layer_name}."
            )

        os.symlink(foreign_layer_path, join(self.path, layer_name))

        # copy the properties of the layer into the properties of this dataset
        layer_properties = self._get_type()(
            foreign_layer_path.parent
        ).properties.data_layers[layer_name]
        self.properties.data_layers[layer_name] = layer_properties
        self.properties._export_as_json()

        self._layers[layer_name] = self._create_layer(
            layer_name,
            _dtype_per_layer_to_dtype_per_channel(
                layer_properties.element_class, layer_properties.num_channels
            ),
            layer_properties.num_channels,
        )
        for resolution in layer_properties.wkw_magnifications:
            self.get_layer(layer_name)._setup_mag(resolution.mag.to_layer_name())
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
        """
        Returns a view of the specified `wkcuber.api.MagDataset.MagDataset`.
        This is a shorthand for `dataset.get_layer(layer_name).get_mag(mag).get_view(...)`

        See `wkcuber.api.MagDataset.get_view` for more details.
        """
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
        """
        Copies the data from the current dataset to `empty_target_ds`. The types of the two datasets can differ
        (e.g. on dataset can be `WKDataset` and the other can be `TiffDataset`).
        Therefore, this method can be used to convert from one type to the other.
        """
        assert (
            len(empty_target_ds.layers) == 0
        ), "Copying dataset failed. The target dataset must be empty."
        with get_executor_for_args(args) as executor:
            for layer_name, layer in self.layers.items():
                largest_segment_id = None
                if (
                    self.properties.data_layers[layer_name].category
                    == Layer.SEGMENTATION_TYPE
                ):
                    largest_segment_id = cast(
                        SegmentationLayerProperties,
                        self.properties.data_layers[layer_name],
                    ).largest_segment_id
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
                        work_on_chunk=_copy_job,
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
        """
        Creates a new `WKDataset` at `new_dataset_path` and copies the data from this dataset to the new dataset.

        This is a shorthand for creating an empty `WKDataset` and then calling `AbstractDataset.copy_dataset`
        """
        new_dataset_path = Path(new_dataset_path)
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
        """
        Creates a new `TiffDataset` at `new_dataset_path` and copies the data from this dataset to the new dataset.

        This is a shorthand for creating an empty `TiffDataset` and then calling `AbstractDataset.copy_dataset`
        """
        new_dataset_path = Path(new_dataset_path)
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
        """
        Creates a new `TiledTiffDataset` at `new_dataset_path` and copies the data from this dataset to the new dataset.

        This is a shorthand for creating an empty `TiledTiffDataset` and then calling `AbstractDataset.copy_dataset`
        """
        new_dataset_path = Path(new_dataset_path)
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
    """
    A dataset is the entry point of the Dataset API. An existing dataset on disk can be opened
    or new datasets can be created.

    A `WKDataset` stores the data in `.wkw` files on disk.
    """

    @classmethod
    def create(
        cls, dataset_path: Union[str, Path], scale: Tuple[float, float, float]
    ) -> "WKDataset":
        """
        Creates a new dataset and the associated `datasource-properties.json`.
        """
        dataset_path = Path(dataset_path)
        name = basename(normpath(dataset_path))
        properties = WKProperties(dataset_path / Properties.FILE_NAME, name, scale)
        return cast(WKDataset, WKDataset._create_with_properties(properties))

    @classmethod
    def get_or_create(
        cls, dataset_path: Union[str, Path], scale: Tuple[float, float, float]
    ) -> "WKDataset":
        """
        Creates a new `WKDataset`, in case it did not exist before, and then returns it.
        The `datasource-properties.json` is used to check if the dataset already exist.
        """
        dataset_path = Path(dataset_path)
        if (
            dataset_path / Properties.FILE_NAME
        ).exists():  # use the properties file to check if the Dataset exists
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
    """
    A dataset is the entry point of the Dataset API. An existing dataset on disk can be opened
    or new datasets can be created.

    A `TiffDataset` stores the data in tiff-files on disk. Each z-slice is stored in a separate tiff-image.
    """

    properties: TiffProperties

    @classmethod
    def create(
        cls,
        dataset_path: Union[str, Path],
        scale: Tuple[float, float, float],
        pattern: Optional[str] = None,
    ) -> "TiffDataset":
        """
        Creates a new dataset and the associated `datasource-properties.json`.
        The `pattern` defines the format of the file structure / filename of the files on disk.
        The default pattern is `"{zzzzz}.tif"`.
        """
        dataset_path = Path(dataset_path)
        if pattern is None:
            pattern = "{zzzzz}.tif"
        _validate_pattern(pattern)
        name = dataset_path.name
        properties = TiffProperties(
            dataset_path / "datasource-properties.json",
            name,
            scale,
            pattern=pattern,
            tile_size=None,
        )
        return cast(TiffDataset, TiffDataset._create_with_properties(properties))

    @classmethod
    def get_or_create(
        cls,
        dataset_path: Union[str, Path],
        scale: Tuple[float, float, float],
        pattern: str = None,
    ) -> "TiffDataset":
        """
        Creates a new `TiffDataset`, in case it did not exist before, and then returns it.
        The `datasource-properties.json` is used to check if the dataset already exist.

        See `TiffDataset.create` for more information.
        """
        dataset_path = Path(dataset_path)
        if (dataset_path / Properties.FILE_NAME).exists():
            # use the properties file to check if the Dataset exists
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
    """
    A dataset is the entry point of the Dataset API. An existing dataset on disk can be opened
    or new datasets can be created.

    A `TiledTiffDataset` stores the data in tiff-files on disk.
    Each z-slice is composed into multiple smaller tiff-image.
    """

    properties: TiffProperties

    @classmethod
    def create(
        cls,
        dataset_path: Union[str, Path],
        scale: Tuple[float, float, float],
        tile_size: Tuple[int, int],
        pattern: Optional[str] = None,
    ) -> "TiledTiffDataset":
        """
        Creates a new dataset and the associated `datasource-properties.json`.
        The `pattern` defines the format of the file structure / filename of the files on disk.
        The default pattern is `"{xxxxx}/{yyyyy}/{zzzzz}.tif"`.
        The `tile_size` specifies the dimensions of a single tiff-tile.
        """
        dataset_path = Path(dataset_path)
        if pattern is None:
            pattern = "{xxxxx}/{yyyyy}/{zzzzz}.tif"
        _validate_pattern(pattern)
        name = dataset_path.name
        properties = TiffProperties(
            dataset_path / "datasource-properties.json",
            name,
            scale,
            pattern=pattern,
            tile_size=tile_size,
        )
        return cast(
            TiledTiffDataset, TiledTiffDataset._create_with_properties(properties)
        )

    @classmethod
    def get_or_create(
        cls,
        dataset_path: Union[str, Path],
        scale: Tuple[float, float, float],
        tile_size: Tuple[int, int],
        pattern: str = None,
    ) -> "TiledTiffDataset":
        """
        Creates a new `TiledTiffDataset`, in case it did not exist before, and then returns it.
        The `datasource-properties.json` is used to check if the dataset already exist.

        See `TiledTiffDataset.create` for more information.
        """
        dataset_path = Path(dataset_path)
        if (dataset_path / Properties.FILE_NAME).exists():
            # use the properties file to check if the Dataset exists
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


def _validate_pattern(pattern: str) -> None:
    assert pattern.count("{") > 0 and pattern.count("}") > 0, (
        f"The provided pattern {pattern} is invalid."
        + " It needs to contain at least one '{' and one '}'."
    )
    assert pattern.count("{") == pattern.count("}"), (
        f"The provided pattern {pattern} is invalid."
        + " The number of '{' does not match the number of '}'."
    )
