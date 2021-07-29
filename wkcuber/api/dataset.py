import operator
import shutil
from argparse import Namespace
from shutil import rmtree
from os import makedirs
from os.path import join, normpath, basename
from pathlib import Path
from typing import Tuple, Union, Dict, Any, Optional, cast

import numpy as np
import os
import re

import wkw

from wkcuber.api.properties.layer_properties import (
    properties_floating_type_to_python_type,
    SegmentationLayerProperties,
    LayerProperties,
)
from wkcuber.api.bounding_box import BoundingBox
from wkcuber.utils import get_executor_for_args, _snake_to_camel_case

from wkcuber.api.properties.dataset_properties import Properties
from wkcuber.api.layer import Layer, LayerCategories, SegmentationLayer
from wkcuber.api.view import View

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


class Dataset:
    """
    A dataset is the entry point of the Dataset API. An existing dataset on disk can be opened
    or new datasets can be created.

    A `Dataset` stores the data in `.wkw` files on disk.

    ## Examples

    ### Creating Datasets
    ```python
    from wkcuber.api.dataset import Dataset

    dataset = Dataset.create(<path_to_new_dataset>, scale=(1, 1, 1))
    # Adds a new layer
    layer = dataset.add_layer(
        layer_name="color",
        category=LayerCategories.COLOR_TYPE,
        dtype_per_channel="uint8",
        num_channels=3
    )
    # Adds an existing layer from a different dataset
    sym_layer = dataset.add_symlink_layer(<foreign_layer_path>)
    ```

    ### Opening Datasets
    ```python
    from wkcuber.api.dataset import Dataset

    dataset = Dataset(<path_to_dataset>)
    # Assuming that the dataset has a layer called 'color'
    layer = dataset.get_layer("color")
    ```

    ### Copying Datasets
    ```python
    from wkcuber.api.dataset import Dataset

    dataset = Dataset(<path_to_dataset>)
    # Copying the dataset with different block_len and file_len
    copy_of_dataset = dataset.copy_dataset(
        <path_to_new_dataset>,
        block_len=8,
        file_len=8
    )
    ```

    ## Functions
    """

    def __init__(self, dataset_path: Union[str, Path]) -> None:
        """
        To open an existing dataset on disk, simply call the constructor of `Dataset`.
        This requires that the `datasource-properties.json` exists. Based on the `datasource-properties.json`,
        a dataset object is constructed. Only layers and magnifications that are listed in the properties are loaded
        (even though there might exists more layer or magnifications on disk).

        The `dataset_path` refers to the top level directory of the dataset (excluding layer or magnification names).
        """

        self.path = Path(dataset_path)
        """Location of the dataset"""

        self.properties: Properties = Properties._from_json(
            self.path / Properties.FILE_NAME
        )
        """
        The metadata from the `datasource-properties.json`. 
        The properties are exported to disk automatically, every time the metadata changes.
        """

        self._layers: Dict[str, Layer] = {}
        self._data_format = "wkw"

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
    def layers(self) -> Dict[str, Layer]:
        """
        Getter for dictionary containing all layers.
        """
        return self._layers

    @classmethod
    def _create_with_properties(cls, properties: Properties) -> "Dataset":
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

    def get_layer(self, layer_name: str) -> Layer:
        """
        Returns the layer called `layer_name` of this dataset. The return type is `wkcuber.api.layer.Layer`.

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
    ) -> Layer:
        """
        Creates a new layer called `layer_name` and adds it to the dataset.
        The dtype can either be specified per layer or per channel.
        If neither of them are specified, `uint8` per channel is used as default.
        When creating a "Segmentation Layer" (`category="segmentation"`),
        the parameter `largest_segment_id` also has to be specified.

        Creates the folder `layer_name` in the directory of `self.path`.

        The return type is `wkcuber.api.layer.Layer`.

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
            layer_name, dtype_per_channel, num_channels, category
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
    ) -> Layer:
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

    def get_segmentation_layer(self) -> SegmentationLayer:
        """
        Returns the only segmentation layer.

        Fails with a IndexError if there are multiple segmentation layers or none.
        """
        return cast(
            SegmentationLayer,
            self._get_layer_by_category(LayerCategories.SEGMENTATION_TYPE),
        )

    def get_color_layer(self) -> Layer:
        """
        Returns the only color layer.

        Fails with a RuntimeError if there are multiple color layers or none.
        """
        return self._get_layer_by_category(LayerCategories.COLOR_TYPE)

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

    def add_symlink_layer(
        self, foreign_layer_path: Union[str, Path], make_relative: bool = False
    ) -> Layer:
        """
        Creates a symlink to the data at `foreign_layer_path` which belongs to another dataset.
        The relevant information from the `datasource-properties.json` of the other dataset is copied to this dataset.
        Note: If the other dataset modifies its bounding box afterwards, the change does not affect this properties
        (or vice versa).
        If make_relative is True, the symlink is made relative to the current dataset path.
        """
        foreign_layer_path = Path(os.path.abspath(foreign_layer_path))
        layer_name = foreign_layer_path.name
        if layer_name in self.layers.keys():
            raise IndexError(
                f"Cannot create symlink to {foreign_layer_path}. This dataset already has a layer called {layer_name}."
            )

        foreign_layer_symlink_path = (
            Path(os.path.relpath(foreign_layer_path, self.path))
            if make_relative
            else foreign_layer_path
        )
        os.symlink(foreign_layer_symlink_path, join(self.path, layer_name))

        # copy the properties of the layer into the properties of this dataset
        layer_properties = Dataset(foreign_layer_path.parent).properties.data_layers[
            layer_name
        ]
        self.properties.data_layers[layer_name] = layer_properties
        self.properties._export_as_json()

        self._layers[layer_name] = self._create_layer(
            layer_name,
            _dtype_per_layer_to_dtype_per_channel(
                layer_properties.element_class, layer_properties.num_channels
            ),
            layer_properties.num_channels,
            layer_properties.category,
        )
        for resolution in layer_properties.wkw_magnifications:
            self.get_layer(layer_name)._setup_mag(resolution.mag.to_layer_name())
        return self.layers[layer_name]

    def add_copy_layer(self, foreign_layer_path: Union[str, Path]) -> Layer:
        """
        Copies the data at `foreign_layer_path` which belongs to another dataset to the current dataset.
        Additionally, the relevant information from the `datasource-properties.json` of the other dataset are copied too.
        """

        foreign_layer_path = Path(os.path.abspath(foreign_layer_path))
        layer_name = foreign_layer_path.name
        if layer_name in self.layers.keys():
            raise IndexError(
                f"Cannot copy {foreign_layer_path}. This dataset already has a layer called {layer_name}."
            )

        shutil.copytree(foreign_layer_path, join(self.path, layer_name))

        # copy the properties of the layer into the properties of this dataset
        layer_properties = Dataset(foreign_layer_path.parent).properties.data_layers[
            layer_name
        ]
        self.properties.data_layers[layer_name] = layer_properties
        self.properties._export_as_json()

        self._layers[layer_name] = self._create_layer(
            layer_name,
            _dtype_per_layer_to_dtype_per_channel(
                layer_properties.element_class, layer_properties.num_channels
            ),
            layer_properties.num_channels,
            layer_properties.category,
        )
        for resolution in layer_properties.wkw_magnifications:
            self.get_layer(layer_name)._setup_mag(resolution.mag.to_layer_name())
        return self.layers[layer_name]

    def copy_dataset(
        self,
        new_dataset_path: Union[str, Path],
        scale: Optional[Tuple[float, float, float]] = None,
        block_len: int = None,
        file_len: int = None,
        compress: Optional[bool] = None,
        args: Optional[Namespace] = None,
    ) -> "Dataset":
        """
        Creates a new dataset at `new_dataset_path` and copies the data from the current dataset to `empty_target_ds`.
        If not specified otherwise, the `scale`, `block_len`, `file_len` and `block_type` of the current dataset are also used for the new dataset.
        """

        new_dataset_path = Path(new_dataset_path)
        if scale is None:
            scale = self.properties.scale
        new_ds = Dataset.create(new_dataset_path, scale=scale)

        with get_executor_for_args(args) as executor:
            for layer_name, layer in self.layers.items():
                largest_segment_id = None
                if (
                    self.properties.data_layers[layer_name].category
                    == LayerCategories.SEGMENTATION_TYPE
                ):
                    largest_segment_id = cast(
                        SegmentationLayerProperties,
                        self.properties.data_layers[layer_name],
                    ).largest_segment_id
                target_layer = new_ds.add_layer(
                    layer_name,
                    self.properties.data_layers[layer_name].category,
                    dtype_per_channel=layer.dtype_per_channel,
                    num_channels=layer.num_channels,
                    largest_segment_id=largest_segment_id,
                )

                bbox = self.properties.get_bounding_box_of_layer(layer_name)

                for mag, mag_view in layer.mags.items():
                    block_len = (
                        block_len
                        if block_len is not None
                        else mag_view.header.block_len
                    )
                    compress = (
                        compress
                        if compress is not None
                        else mag_view.header.block_type != wkw.Header.BLOCK_TYPE_RAW
                    )
                    file_len = (
                        file_len if file_len is not None else mag_view.header.file_len
                    )
                    target_mag = target_layer.add_mag(
                        mag, block_len, file_len, compress
                    )

                    # The bounding box needs to be updated manually because chunked views do not have a reference to the dataset itself
                    # The base view of a MagDataset always starts at (0, 0, 0)
                    target_mag.global_offset = (0, 0, 0)
                    target_mag.size = cast(
                        Tuple[int, int, int],
                        tuple(
                            BoundingBox(topleft=bbox[0], size=bbox[1])
                            .align_with_mag(mag, ceil=True)
                            .in_mag(mag)
                            .bottomright
                        ),
                    )
                    target_mag.layer.set_bounding_box(offset=bbox[0], size=bbox[1])

                    # The data gets written to the target_mag.
                    # Therefore, the chunk size is determined by the target_mag to prevent concurrent writes
                    mag_view.for_zipped_chunks(
                        work_on_chunk=_copy_job,
                        target_view=target_mag.get_view(),
                        source_chunk_size=target_mag._get_file_dimensions(),
                        target_chunk_size=target_mag._get_file_dimensions(),
                        executor=executor,
                    )
        return new_ds

    def _get_layer_by_category(self, category: str) -> Layer:
        assert (
            category == LayerCategories.COLOR_TYPE
            or category == LayerCategories.SEGMENTATION_TYPE
        )
        layer_property_type = (
            SegmentationLayerProperties
            if category == LayerCategories.SEGMENTATION_TYPE
            else LayerProperties
        )

        layer_properties = [
            layer_property
            for layer_property in self.properties.data_layers.values()
            if type(layer_property) == layer_property_type
        ]

        if len(layer_properties) == 1:
            return self.get_layer(layer_properties[0].name)
        elif len(layer_properties) == 0:
            raise IndexError(
                f"Failed to get segmentation layer: There is no {category} layer."
            )
        else:
            raise IndexError(
                f"Failed to get segmentation layer: There are multiple {category} layer."
            )

    @property
    def name(self) -> str:
        return self.properties._name

    @name.setter
    def name(self, name: str) -> None:
        self.properties._name = name
        self.properties._export_as_json()

    @classmethod
    def create(
        cls,
        dataset_path: Union[str, Path],
        scale: Tuple[float, float, float],
        name: Optional[str] = None,
    ) -> "Dataset":
        """
        Creates a new dataset and the associated `datasource-properties.json`.
        """
        dataset_path = Path(dataset_path)
        name = name if name is not None else basename(normpath(dataset_path))
        properties = Properties(dataset_path / Properties.FILE_NAME, name, scale)
        return Dataset._create_with_properties(properties)

    @classmethod
    def get_or_create(
        cls,
        dataset_path: Union[str, Path],
        scale: Tuple[float, float, float],
        name: Optional[str] = None,
    ) -> "Dataset":
        """
        Creates a new `Dataset`, in case it did not exist before, and then returns it.
        The `datasource-properties.json` is used to check if the dataset already exist.
        """
        dataset_path = Path(dataset_path)
        if (
            dataset_path / Properties.FILE_NAME
        ).exists():  # use the properties file to check if the Dataset exists
            ds = Dataset(dataset_path)
            assert tuple(ds.properties.scale) == tuple(
                scale
            ), f"Cannot get_or_create Dataset: The dataset {dataset_path} already exists, but the scales do not match ({ds.properties.scale} != {scale})"
            if name is not None:
                assert (
                    ds.name == name
                ), f"Cannot get_or_create Dataset: The dataset {dataset_path} already exists, but the names do not match ({ds.name} != {name})"
            return ds
        else:
            return cls.create(dataset_path, scale, name)

    def _create_layer(
        self,
        layer_name: str,
        dtype_per_channel: np.dtype,
        num_channels: int,
        category: str,
    ) -> Layer:
        layer_type = (
            Layer if category == LayerCategories.COLOR_TYPE else SegmentationLayer
        )
        return layer_type(layer_name, self, dtype_per_channel, num_channels)

    def set_view_configuration(
        self, view_configuration: "DatasetViewConfiguration"
    ) -> None:
        self.properties._default_view_configuration = {
            _snake_to_camel_case(k): v
            for k, v in vars(view_configuration).items()
            if v is not None
        }
        self.properties._export_as_json()  # update properties on disk

    def get_view_configuration(self) -> Optional["DatasetViewConfiguration"]:
        view_configuration_dict = self.properties.default_view_configuration
        if view_configuration_dict is None:
            return None

        return DatasetViewConfiguration(
            four_bit=view_configuration_dict.get("fourBit"),
            interpolation=view_configuration_dict.get("interpolation"),
            render_missing_data_black=view_configuration_dict.get(
                "renderMissingDataBlack"
            ),
            loading_strategy=view_configuration_dict.get("loadingStrategy"),
            segmentation_pattern_opacity=view_configuration_dict.get(
                "segmentationPatternOpacity"
            ),
            zoom=view_configuration_dict.get("zoom"),
            position=cast(
                Tuple[int, int, int], tuple(view_configuration_dict["position"])
            )
            if "position" in view_configuration_dict
            else None,
            rotation=cast(
                Tuple[int, int, int], tuple(view_configuration_dict["rotation"])
            )
            if "rotation" in view_configuration_dict
            else None,
        )

    def __repr__(self) -> str:
        return repr("Dataset(%s)" % self.path)


class DatasetViewConfiguration:
    """
    Stores information on how the dataset is shown in webknossos by default.
    """

    def __init__(
        self,
        four_bit: Optional[bool] = None,
        interpolation: Optional[bool] = None,
        render_missing_data_black: Optional[bool] = None,
        loading_strategy: Optional[str] = None,
        segmentation_pattern_opacity: Optional[int] = None,
        zoom: Optional[float] = None,
        position: Optional[Tuple[int, int, int]] = None,
        rotation: Optional[Tuple[int, int, int]] = None,
    ):
        self.four_bit = four_bit
        self.interpolation = interpolation
        self.render_missing_data_black = render_missing_data_black
        self.loading_strategy = loading_strategy
        self.segmentation_pattern_opacity = segmentation_pattern_opacity
        self.zoom = zoom
        self.position = position
        self.rotation = rotation
