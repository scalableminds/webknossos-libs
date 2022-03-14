from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import attr
import cattr
import numpy as np
from cattr.gen import make_dict_structure_fn, make_dict_unstructure_fn, override

from webknossos.geometry import BoundingBox, Mag, Vec3Int
from webknossos.utils import snake_to_camel_case, warn_deprecated

from ._array import ArrayException, BaseArray, DataFormat
from .layer_categories import LayerCategoryType


def _extract_num_channels(
    num_channels_in_properties: Optional[int],
    path: Path,
    layer: str,
    mag: Optional[Union[int, Mag]],
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


_properties_floating_type_to_python_type: Dict[Union[str, type], np.dtype] = {
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

    four_bit: Optional[bool] = None
    interpolation: Optional[bool] = None
    render_missing_data_black: Optional[bool] = None
    loading_strategy: Optional[str] = None
    segmentation_pattern_opacity: Optional[int] = None
    zoom: Optional[float] = None
    position: Optional[Tuple[int, int, int]] = None
    rotation: Optional[Tuple[int, int, int]] = None


@attr.define
class LayerViewConfiguration:
    """
    Stores information on how the dataset is shown in webknossos by default.
    """

    color: Optional[Tuple[int, int, int]] = None
    alpha: Optional[float] = None
    intensity_range: Optional[Tuple[float, float]] = None
    min: Optional[float] = None  # pylint: disable=redefined-builtin
    max: Optional[float] = None  # pylint: disable=redefined-builtin
    is_disabled: Optional[bool] = None
    is_inverted: Optional[bool] = None
    is_in_edit_mode: Optional[bool] = None


# --- Property --------------------


@attr.define
class MagViewProperties:
    mag: Mag
    cube_length: Optional[int] = None

    @property
    def resolution(self) -> Mag:
        warn_deprecated("resolution", "mag")
        return self.mag


@attr.define
class LayerProperties:
    name: str
    category: LayerCategoryType
    bounding_box: BoundingBox
    element_class: str
    data_format: DataFormat
    mags: List[MagViewProperties]
    num_channels: Optional[int] = None
    default_view_configuration: Optional[LayerViewConfiguration] = None

    @property
    def resolutions(self) -> List[MagViewProperties]:
        warn_deprecated("resolutions", "mags")
        return self.mags


@attr.define
class SegmentationLayerProperties(LayerProperties):
    largest_segment_id: int = -1
    mappings: List[str] = []


@attr.define
class DatasetProperties:
    id: Dict[str, str]
    scale: Tuple[float, float, float]
    data_layers: List[
        Union[
            SegmentationLayerProperties,
            LayerProperties,
        ]
    ]
    default_view_configuration: Optional[DatasetViewConfiguration] = None


# --- Converter --------------------

dataset_converter = cattr.Converter()

# register (un-)structure hooks for non-attr-classes
bbox_to_wkw: Callable[[BoundingBox], dict] = lambda o: o.to_wkw_dict()
dataset_converter.register_unstructure_hook(BoundingBox, bbox_to_wkw)
dataset_converter.register_structure_hook(
    BoundingBox, lambda d, _: BoundingBox.from_wkw_dict(d)
)


def mag_unstructure(mag: Mag) -> List[int]:
    return mag.to_list()


dataset_converter.register_unstructure_hook(Mag, mag_unstructure)
dataset_converter.register_structure_hook(Mag, lambda d, _: Mag(d))

vec3int_to_array: Callable[[Vec3Int], List[int]] = lambda o: o.to_list()
dataset_converter.register_unstructure_hook(Vec3Int, vec3int_to_array)
dataset_converter.register_structure_hook(
    Vec3Int, lambda d, _: Vec3Int.full(d) if isinstance(d, int) else Vec3Int(d)
)

dataset_converter.register_structure_hook(LayerCategoryType, lambda d, _: str(d))

# Register (un-)structure hooks for attr-classes to bring the data into the expected format.
# The properties on disk (in datasource-properties.json) use camel case for the names of the attributes.
# However, we use snake case for the attribute names in python.
# This requires that the names of the attributes are renamed during (un-)structuring.
# Additionally we only want to unstructure attributes which don't have the default value
# (e.g. Layer.default_view_configuration has many attributes which are all optionally).
for cls in [
    DatasetProperties,
    MagViewProperties,
    DatasetViewConfiguration,
    LayerViewConfiguration,
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
                for a in attr.fields(cls)  # pylint: disable=not-an-iterable
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
                for a in attr.fields(cls)  # pylint: disable=not-an-iterable
            },
        ),
    )

# The serialization of `LayerProperties` differs slightly based on whether it is a `wkw` or `zarr` layer.
# These post-unstructure and pre-structure functions perform the conditional field renames.
def mag_view_properties_post_structure(d: Dict[str, Any]) -> Dict[str, Any]:
    d["resolution"] = d["mag"]
    del d["mag"]
    return d


def mag_view_properties_pre_unstructure(d: Dict[str, Any]) -> Dict[str, Any]:
    d["mag"] = d["resolution"]
    del d["resolution"]
    return d


def layer_properties_post_unstructure(
    converter_fn: Callable[
        [Union[LayerProperties, SegmentationLayerProperties]], Dict[str, Any]
    ]
) -> Callable[[Union[LayerProperties, SegmentationLayerProperties]], Dict[str, Any]]:
    def __layer_properties_post_unstructure(
        obj: Union[LayerProperties, SegmentationLayerProperties]
    ) -> Dict[str, Any]:
        d = converter_fn(obj)
        if d["dataFormat"] == "wkw":
            d["wkwResolutions"] = [
                mag_view_properties_post_structure(m) for m in d["mags"]
            ]
            del d["mags"]
        return d

    return __layer_properties_post_unstructure


def layer_properties_pre_structure(
    converter_fn: Callable[
        [Dict[str, Any]], Union[LayerProperties, SegmentationLayerProperties]
    ]
) -> Callable[
    [Any, Type[Union[LayerProperties, SegmentationLayerProperties]]],
    Union[LayerProperties, SegmentationLayerProperties],
]:
    def __layer_properties_pre_structure(
        d: Dict[str, Any],
        _type: Type[Union[LayerProperties, SegmentationLayerProperties]],
    ) -> Union[LayerProperties, SegmentationLayerProperties]:
        if d["dataFormat"] == "wkw":
            d["mags"] = [
                mag_view_properties_pre_unstructure(m) for m in d["wkwResolutions"]
            ]
            del d["wkwResolutions"]
        obj = converter_fn(d)
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
                    for a in attr.fields(cls)  # pylint: disable=not-an-iterable
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
                    for a in attr.fields(cls)  # pylint: disable=not-an-iterable
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
    Union[
        SegmentationLayerProperties,
        LayerProperties,
    ],
    disambiguate_layer_properties,
)
