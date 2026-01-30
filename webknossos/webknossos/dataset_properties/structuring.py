import copy
from collections.abc import Callable
from functools import cache
from typing import Any

import attr
import cattr
import numpy as np
from cattr.gen import make_dict_structure_fn, make_dict_unstructure_fn, override

from ..dataset_properties import (
    AttachmentProperties,
    AttachmentsProperties,
    DatasetProperties,
    DatasetViewConfiguration,
    LayerProperties,
    LayerViewConfiguration,
    LengthUnit,
    MagViewProperties,
    SegmentationLayerProperties,
    length_unit_from_str,
)
from ..dataset_properties.dataset_properties import DEFAULT_LENGTH_UNIT_STR
from ..geometry import Mag, NDBoundingBox, Vec3Int
from ..utils import snake_to_camel_case
from .layer_categories import LayerCategoryType

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


# register (un-)structure hooks for non-attr-classes
bbox_to_wkw: Callable[[NDBoundingBox], dict] = lambda o: o.to_wkw_dict()  # noqa: E731


def mag_unstructure(mag: Mag) -> list[int]:
    return mag.to_list()


vec3int_to_array: Callable[[Vec3Int], list[int]] = lambda o: o.to_list()  # noqa: E731


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


def layer_properties_post_unstructure(
    converter_fn: Callable[
        [LayerProperties | SegmentationLayerProperties], dict[str, Any]
    ],
) -> Callable[[LayerProperties | SegmentationLayerProperties], dict[str, Any]]:
    def __layer_properties_post_unstructure(
        obj: LayerProperties | SegmentationLayerProperties,
    ) -> dict[str, Any]:
        d = converter_fn(obj)

        for mag in d["mags"]:
            if "axisOrder" in d["boundingBox"]:
                mag["axisOrder"] = d["boundingBox"]["axisOrder"]
            if "channelIndex" in d["boundingBox"]:
                mag["channelIndex"] = d["boundingBox"]["channelIndex"]

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
            if "wkwResolutions" in d:
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
            if "channelIndex" in first_mag:
                d["boundingBox"]["channelIndex"] = first_mag["channelIndex"]
            if "axisOrder" in first_mag:
                assert all(
                    first_mag["axisOrder"] == mag["axisOrder"] for mag in d["mags"]
                )
                d["boundingBox"]["axisOrder"] = copy.deepcopy(first_mag["axisOrder"])
                assert all(
                    first_mag["axisOrder"] == mag["axisOrder"] for mag in d["mags"]
                ), "axisOrder must be the same for all mags"

        if "numChannels" in d:
            d["boundingBox"]["numChannels"] = d["numChannels"]

        obj = converter_fn(d, type_value)
        return obj

    return __layer_properties_pre_structure


@cache
def get_dataset_converter() -> cattr.Converter:
    dataset_converter = cattr.Converter()
    dataset_converter.register_unstructure_hook(NDBoundingBox, bbox_to_wkw)
    dataset_converter.register_structure_hook(
        NDBoundingBox, lambda d, _: NDBoundingBox.from_wkw_dict(d)
    )
    dataset_converter.register_unstructure_hook(Mag, mag_unstructure)
    dataset_converter.register_structure_hook(Mag, lambda d, _: Mag(d))

    dataset_converter.register_structure_hook(
        LengthUnit, lambda d, _: length_unit_from_str(d)
    )

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
                a.name: override(
                    omit_if_default=True, rename=snake_to_camel_case(a.name)
                )
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
    return dataset_converter
