import copy
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import (
    Any,
    Literal,
)

import attr
import cattr
import numpy as np
from cattr.gen import make_dict_structure_fn, make_dict_unstructure_fn, override

from ..geometry import Mag, NDBoundingBox, Vec3Int
from ..utils import snake_to_camel_case
from ._array import ArrayException, BaseArray
from .data_format import AttachmentDataFormat, DataFormat
from .layer_categories import LayerCategoryType
from .length_unit import (
    _LENGTH_UNIT_TO_NANOMETER,
    LengthUnit,
    length_unit_from_str,
)

DEFAULT_LENGTH_UNIT = LengthUnit.NANOMETER
DEFAULT_LENGTH_UNIT_STR = DEFAULT_LENGTH_UNIT.value


def _extract_num_channels(
    num_channels_in_properties: int | None,
    path: Path,
    layer: str,
    mag: int | Mag | None,
) -> int:
    # if a wk dataset is not created with this API, then it most likely doesn't have the attribute 'numChannels' in the
    # datasource-properties.json. In this case we need to extract the number of channels from the 'header.wkw'.
    if num_channels_in_properties is not None:
        return num_channels_in_properties

    if mag is None:
        # Unable to extract the 'num_channels' from the 'header.wkw' if the dataset has no magnifications.
        # This should never be the case because wkw-datasets that are created without this API always have a magnification.
        raise RuntimeError(
            "Cannot extract the number of channels of a dataset without a properties file and without any magnifications"
        )

    mag = Mag(mag)
    array_file_path = path / layer / mag.to_layer_name()
    try:
        array = BaseArray.open(array_file_path)
    except ArrayException as e:
        raise Exception(
            f"The dataset you are trying to open does not have the attribute 'numChannels' for layer {layer}. "
            f"However, this attribute is necessary. To mitigate this problem, it was tried to locate "
            f"the file {array_file_path} to extract the num_channels from there. "
            f"Since this file does not exist, the attempt to open the dataset failed. "
            f"Please add the attribute manually to solve the problem. "
            f"If the layer does not contain any data, you can also delete the layer and add it again.",
        ) from e
    return array.info.num_channels


def float_tpl(voxel_size: list | tuple) -> Iterable:
    # Fix for mypy bug https://github.com/python/mypy/issues/5313.
    # Solution based on other issue for the same bug: https://github.com/python/mypy/issues/8389.
    return tuple(
        (
            voxel_size[0],
            voxel_size[1],
            voxel_size[2],
        )
    )


_properties_floating_type_to_python_type: dict[str | type, np.dtype] = {
    "float": np.dtype("float32"),
    #  np.float: np.dtype("float32"),  # np.float is an alias for float
    float: np.dtype("float32"),
    "double": np.dtype("float64"),
}

_python_floating_type_to_properties_type = {
    "float32": "float",
    "float64": "double",
}


# --- View configuration --------------------


@attr.define
class DatasetViewConfiguration:
    """
    Stores information on how the dataset is shown in webknossos by default.
    """

    four_bit: bool | None = None
    interpolation: bool | None = None
    render_missing_data_black: bool | None = None
    loading_strategy: str | None = None
    segmentation_pattern_opacity: int | None = None
    zoom: float | None = None
    position: tuple[int, int, int] | None = None
    rotation: tuple[int, int, int] | None = None


@attr.define
class LayerViewConfiguration:
    """
    Stores information on how the dataset is shown in webknossos by default.
    """

    color: tuple[int, int, int] | None = None
    """Color in RGB from 0 to 255. The WEBKNOSSOS default is [255, 255, 255]."""

    alpha: float | None = None
    """Alpha value from 0 to 100. The WEBKNOSSOS default is 100 except for segmentation
    layers where it is 20."""

    intensity_range: tuple[float, float] | None = None
    """Min and max data value range (dependent on the layer's data type). Can be used to threshold the value range.
    The WEBKNOSSOS default is the full value range."""

    min: float | None = None
    """Minimum data value that might be encountered. This will restrict the histogram in WEBKNOSSOS and possibly overwrite
    the min value of the `intensityRange` (if that is lower)."""

    max: float | None = None
    """Maximum data value that might be encountered. This will restrict the histogram in WEBKNOSSOS and possibly overwrite
    the max value of the `intensityRange` (if that is higher)."""

    is_disabled: bool | None = None
    """Disable a layer. The WEBKNOSSOS default is False."""

    is_inverted: bool | None = None
    """Invert a layer. The WEBKNOSSOS default is False."""

    is_in_edit_mode: bool | None = None
    """Enable the histogram edit mode. The WEBKNOSSOS default is False."""

    mapping: dict[str, str] | None = None
    """Enables ID mapping for a segmentation layer and applies the selected mapping by default. The default WK behavior is to disable ID mapping. Expected values is a Dict with {"name": my_mapping_name, "type": "HDF5"}."""


# --- Property --------------------


@attr.define
class MagViewProperties:
    mag: Mag
    path: str | None = None
    cube_length: int | None = None
    axis_order: dict[str, int] | None = None


@attr.define
class AxisProperties:
    name: str
    bounds: tuple[int, int]
    index: int


@attr.define
class LayerProperties:
    name: str
    category: LayerCategoryType
    bounding_box: NDBoundingBox
    element_class: str
    data_format: DataFormat
    mags: list[MagViewProperties]
    num_channels: int | None = None
    default_view_configuration: LayerViewConfiguration | None = None


@attr.define
class AttachmentProperties:
    name: str
    path: str
    data_format: AttachmentDataFormat


@attr.define
class AttachmentsProperties:
    meshes: list[AttachmentProperties] | None = None
    agglomerates: list[AttachmentProperties] | None = None
    segment_index: AttachmentProperties | None = None
    cumsum: AttachmentProperties | None = None
    connectomes: list[AttachmentProperties] | None = None

    def __iter__(self) -> Iterator[AttachmentProperties]:
        for attachment in self.meshes or []:
            yield attachment
        for attachment in self.agglomerates or []:
            yield attachment
        if self.segment_index is not None:
            yield self.segment_index
        if self.cumsum is not None:
            yield self.cumsum
        for attachment in self.connectomes or []:
            yield attachment


@attr.define
class SegmentationLayerProperties(LayerProperties):
    largest_segment_id: int | None = None
    mappings: list[str] = []
    attachments: AttachmentsProperties = attr.field(factory=AttachmentsProperties)


@attr.define
class VoxelSize:
    factor: tuple[float, float, float] = attr.field(converter=float_tpl)
    unit: LengthUnit = DEFAULT_LENGTH_UNIT

    def to_nanometer(self) -> tuple[float, float, float]:
        conversion_factor = _LENGTH_UNIT_TO_NANOMETER[self.unit]
        return (
            self.factor[0] * conversion_factor,
            self.factor[1] * conversion_factor,
            self.factor[2] * conversion_factor,
        )


@attr.define
class DatasetProperties:
    id: dict[str, str]
    scale: VoxelSize
    data_layers: list[SegmentationLayerProperties | LayerProperties]
    version: Literal[1]
    default_view_configuration: DatasetViewConfiguration | None = None

    def update_for_layer(
        self, layer_name: str, layer_properties: LayerProperties
    ) -> None:
        for i, layer in enumerate(self.data_layers):
            if layer.name == layer_name:
                self.data_layers[i] = layer_properties
                return
        raise KeyError(f"Layer {layer_name} not found in the dataset properties.")


# --- Converter --------------------

dataset_converter = cattr.Converter()

# register (un-)structure hooks for non-attr-classes
bbox_to_wkw: Callable[[NDBoundingBox], dict] = lambda o: o.to_wkw_dict()  # noqa: E731
dataset_converter.register_unstructure_hook(NDBoundingBox, bbox_to_wkw)
dataset_converter.register_structure_hook(
    NDBoundingBox, lambda d, _: NDBoundingBox.from_wkw_dict(d)
)


def mag_unstructure(mag: Mag) -> list[int]:
    return mag.to_list()


dataset_converter.register_unstructure_hook(Mag, mag_unstructure)
dataset_converter.register_structure_hook(Mag, lambda d, _: Mag(d))

dataset_converter.register_structure_hook(
    LengthUnit, lambda d, _: length_unit_from_str(d)
)

vec3int_to_array: Callable[[Vec3Int], list[int]] = lambda o: o.to_list()  # noqa: E731
dataset_converter.register_unstructure_hook(Vec3Int, vec3int_to_array)
dataset_converter.register_structure_hook(
    Vec3Int, lambda d, _: Vec3Int.full(d) if isinstance(d, int) else Vec3Int(d)
)

dataset_converter.register_structure_hook_func(
    lambda d: d == LayerCategoryType,  # type: ignore[comparison-overlap]
    lambda d, _: str(d),
)

# Register (un-)structure hooks for attr-classes to bring the data into the expected format.
# The properties on disk (in datasource-properties.json) use camel case for the names of the attributes.
# However, we use snake case for the attribute names in python.
# This requires that the names of the attributes are renamed during (un-)structuring.
# Additionally we only want to unstructure attributes which don't have the default value
# (e.g. Layer.default_view_configuration has many attributes which are all optionally).
for cls in [
    MagViewProperties,
    DatasetViewConfiguration,
    LayerViewConfiguration,
    AttachmentProperties,
    AttachmentsProperties,
]:
    dataset_converter.register_unstructure_hook(
        cls,
        make_dict_unstructure_fn(
            cls,
            dataset_converter,
            **{
                a.name: override(
                    omit_if_default=True, rename=snake_to_camel_case(a.name)
                )
                for a in attr.fields(cls)  # type: ignore
            },
        ),
    )
    dataset_converter.register_structure_hook(
        cls,
        make_dict_structure_fn(
            cls,
            dataset_converter,
            **{
                a.name: override(rename=snake_to_camel_case(a.name))
                for a in attr.fields(cls)  # type: ignore
            },
        ),
    )


def dataset_properties_pre_structure(converter_fn: Callable) -> Callable:
    def __dataset_properties_pre_structure(
        d: dict[str, Any], type_value: type[DatasetProperties]
    ) -> dict[str, Any]:
        if isinstance(d["scale"], list):
            d["scale"] = {"unit": DEFAULT_LENGTH_UNIT_STR, "factor": d["scale"]}
        if "version" not in d:
            d["version"] = 1
        obj = converter_fn(d, type_value)
        return obj

    return __dataset_properties_pre_structure


def layer_properties_post_unstructure(
    converter_fn: Callable[
        [LayerProperties | SegmentationLayerProperties], dict[str, Any]
    ],
) -> Callable[[LayerProperties | SegmentationLayerProperties], dict[str, Any]]:
    def __layer_properties_post_unstructure(
        obj: LayerProperties | SegmentationLayerProperties,
    ) -> dict[str, Any]:
        d = converter_fn(obj)
        if d["dataFormat"] == "wkw":
            d["wkwResolutions"] = [
                mag_view_properties_post_unstructure(m) for m in d["mags"]
            ]
            del d["mags"]

        # json expects nd_bounding_box to be represented as bounding_box and additional_axes
        if "additionalAxes" in d["boundingBox"]:
            d["additionalAxes"] = d["boundingBox"]["additionalAxes"]
            del d["boundingBox"]["additionalAxes"]

        if "attachments" in d:
            if all(p is None or len(p) == 0 for p in d["attachments"].values()):
                del d["attachments"]
        return d

    return __layer_properties_post_unstructure


def layer_properties_pre_structure(
    converter_fn: Callable[
        [dict[str, Any], type[LayerProperties | SegmentationLayerProperties]],
        LayerProperties | SegmentationLayerProperties,
    ],
) -> Callable[
    [Any, type[LayerProperties | SegmentationLayerProperties]],
    LayerProperties | SegmentationLayerProperties,
]:
    def __layer_properties_pre_structure(
        d: dict[str, Any],
        type_value: type[LayerProperties | SegmentationLayerProperties],
    ) -> LayerProperties | SegmentationLayerProperties:
        if d["dataFormat"] == "wkw":
            d["mags"] = [
                mag_view_properties_pre_structure(m) for m in d["wkwResolutions"]
            ]
            del d["wkwResolutions"]
        # bounding_box and additional_axes are internally handled as nd_bounding_box
        if "additionalAxes" in d:
            d["boundingBox"]["additionalAxes"] = copy.deepcopy(d["additionalAxes"])
            del d["additionalAxes"]
        if len(d["mags"]) > 0:
            first_mag = d["mags"][0]
            if "axisOrder" in first_mag:
                assert first_mag["axisOrder"]["c"] == 0, (
                    "The channels c must have index 0 in axis order."
                )
                assert all(
                    first_mag["axisOrder"] == mag["axisOrder"] for mag in d["mags"]
                )
                d["boundingBox"]["axisOrder"] = copy.deepcopy(first_mag["axisOrder"])
                del d["boundingBox"]["axisOrder"]["c"]

        obj = converter_fn(d, type_value)
        return obj

    return __layer_properties_pre_structure


for cls in [
    LayerProperties,
    SegmentationLayerProperties,
]:
    dataset_converter.register_unstructure_hook(
        cls,
        layer_properties_post_unstructure(
            make_dict_unstructure_fn(
                cls,
                dataset_converter,
                **{
                    a.name: override(
                        omit_if_default=True, rename=snake_to_camel_case(a.name)
                    )
                    for a in attr.fields(cls)  # type: ignore
                },
            )
        ),
    )
    dataset_converter.register_structure_hook(
        cls,
        layer_properties_pre_structure(
            make_dict_structure_fn(
                cls,
                dataset_converter,
                **{
                    a.name: override(rename=snake_to_camel_case(a.name))
                    for a in attr.fields(cls)  # type: ignore
                },
            )
        ),
    )


# Disambiguation of Unions only work automatically if the two attrs-classes have at least 1 unique attribute
# This is not the case here because SegmentationLayerProperties inherits LayerProperties
def disambiguate_layer_properties(obj: dict, _: Any) -> LayerProperties:
    if obj["category"] == "color":
        return dataset_converter.structure(obj, LayerProperties)
    elif obj["category"] == "segmentation":
        return dataset_converter.structure(obj, SegmentationLayerProperties)
    else:
        raise RuntimeError(
            "Failed to read the properties of a layer: the category has to be `color` or `segmentation`."
        )


dataset_converter.register_structure_hook(
    SegmentationLayerProperties | LayerProperties,
    disambiguate_layer_properties,
)


dataset_converter.register_unstructure_hook(
    DatasetProperties,
    make_dict_unstructure_fn(
        DatasetProperties,
        dataset_converter,
        **{
            a.name: override(omit_if_default=True, rename=snake_to_camel_case(a.name))
            for a in attr.fields(DatasetProperties)
        },  # type: ignore[arg-type]
    ),
)
dataset_converter.register_structure_hook(
    DatasetProperties,
    dataset_properties_pre_structure(
        make_dict_structure_fn(
            DatasetProperties,
            dataset_converter,
            **{
                a.name: override(rename=snake_to_camel_case(a.name))
                for a in attr.fields(DatasetProperties)
            },  # type: ignore[arg-type]
        )
    ),
)


# The serialization of `LayerProperties` differs slightly based on whether it is a `wkw` or `zarr` layer.
# These post-unstructure and pre-structure functions perform the conditional field renames.
def mag_view_properties_post_unstructure(d: dict[str, Any]) -> dict[str, Any]:
    d["resolution"] = d["mag"]
    del d["mag"]
    return d


def mag_view_properties_pre_structure(d: dict[str, Any]) -> dict[str, Any]:
    d["mag"] = d["resolution"]
    del d["resolution"]
    return d
