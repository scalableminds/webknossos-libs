from collections.abc import Iterable, Iterator

import attr

from ..geometry import Mag, NDBoundingBox
from .data_format import AttachmentDataFormat, DataFormat
from .layer_categories import LayerCategoryType
from .length_unit import _LENGTH_UNIT_TO_NANOMETER, LengthUnit

DEFAULT_LENGTH_UNIT = LengthUnit.NANOMETER
DEFAULT_LENGTH_UNIT_STR = DEFAULT_LENGTH_UNIT.value


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


@attr.define
class MagViewProperties:
    mag: Mag
    path: str | None = None
    """
    Could be None for older datasource-proterties.json files.
    """
    cube_length: int | None = None


@attr.define
class LayerProperties:
    name: str
    category: LayerCategoryType
    bounding_box: NDBoundingBox
    element_class: str
    data_format: DataFormat
    mags: list[MagViewProperties]
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
    """
    id is a legacy field that is not used anymore. Its keys are name (dataset directory name) and team (organization id)
    However, webknossos will take both from the dataset path and not from what is here in the datasource-properties.json.
    """
    scale: VoxelSize
    data_layers: list[SegmentationLayerProperties | LayerProperties]
    version: int
    """
    A default version is set during structuring
    """
    default_view_configuration: DatasetViewConfiguration | None = None

    def update_for_layer(
        self, layer_name: str, layer_properties: LayerProperties
    ) -> None:
        for i, layer in enumerate(self.data_layers):
            if layer.name == layer_name:
                self.data_layers[i] = layer_properties
                return
        raise KeyError(f"Layer {layer_name} not found in the dataset properties.")
