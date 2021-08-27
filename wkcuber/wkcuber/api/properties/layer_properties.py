import warnings
from os.path import join, dirname, isfile
from pathlib import Path
from typing import Tuple, Type, Union, Any, Dict, List, Optional, cast
import numpy as np

from wkw import wkw

from wkcuber.api.properties.resolution_properties import Resolution
from wkcuber.mag import Mag
from wkcuber.api.bounding_box import BoundingBox


def _extract_num_channels(
    num_channels_in_properties: Optional[int],
    path: Path,
    layer: str,
    mag: Optional[Dict[str, int]],
) -> int:
    # if a wk dataset is not created with this API, then it most likely doesn't have the attribute 'num_channels' in the
    # datasource-properties.json. In this case we need to extract the 'num_channels' from the 'header.wkw'.
    if num_channels_in_properties is not None:
        return num_channels_in_properties

    if mag is None:
        # Unable to extract the 'num_channels' from the 'header.wkw' if the dataset has no magnifications.
        # This should never be the case because wkw-datasets that are created without this API always have a magnification.
        raise RuntimeError(
            "Cannot extract the number of channels of a dataset without a properties file and without any magnifications"
        )

    wkw_ds_file_path = join(
        dirname(path), layer, Mag(mag["resolution"]).to_layer_name()
    )
    if not isfile(join(wkw_ds_file_path, "header.wkw")):
        raise Exception(
            f"The dataset you are trying to open does not have the attribute 'num_channels' for layer {layer}. "
            f"However, this attribute is necessary. To mitigate this problem, it was tried to locate "
            f"the file {wkw_ds_file_path} to extract the num_channels from there. "
            f"Since this file does not exist, the attempt to open the dataset failed."
            f"Please add the attribute manually to solve the problem. "
            f"If the layer does not contain any data, you can also delete the layer and add it again."
        )
    wkw_ds = wkw.Dataset.open(wkw_ds_file_path)
    return wkw_ds.header.num_channels


properties_floating_type_to_python_type = {
    "float": np.dtype("float32"),
    np.float: np.dtype("float32"),
    float: np.dtype("float32"),
    "double": np.dtype("float64"),
}

python_floating_type_to_properties_type = {
    "float32": "float",
    "float64": "double",
}


class LayerProperties:
    def __init__(
        self,
        name: str,
        category: str,
        element_class: str,
        data_format: str,
        num_channels: int,
        bounding_box: Dict[str, Union[int, Tuple[int, int, int]]] = None,
        resolutions: List[Resolution] = None,
        default_view_configuration: Optional[dict] = None,
    ):
        self._name = name
        self._category = category
        self._element_class = element_class
        self._data_format = data_format
        self._num_channels = num_channels
        self._bounding_box = bounding_box or {
            "topLeft": (-1, -1, -1),
            "width": 0,
            "height": 0,
            "depth": 0,
        }
        self._wkw_magnifications: List[Resolution] = resolutions or []
        self._default_view_configuration = default_view_configuration

    def _to_json(self) -> Dict[str, Any]:
        layer_properties = {
            "name": self.name,
            "category": self.category,
            "elementClass": python_floating_type_to_properties_type.get(
                self.element_class, self.element_class
            ),
            "dataFormat": self._data_format,
            "num_channels": self.num_channels,
            "boundingBox": {}
            if self.bounding_box is None
            else {
                "topLeft": self.bounding_box["topLeft"],
                "width": self.bounding_box["width"],
                "height": self.bounding_box["height"],
                "depth": self.bounding_box["depth"],
            },
            "wkwResolutions": [r._to_json() for r in self.wkw_magnifications],
        }
        if self.default_view_configuration is not None:
            layer_properties[
                "defaultViewConfiguration"
            ] = self.default_view_configuration

        return layer_properties

    @classmethod
    def _from_json(
        cls,
        json_data: Dict[str, Any],
        resolution_type: Type[Resolution],
        dataset_path: Path,
    ) -> "LayerProperties":
        # create LayerProperties without resolutions
        layer_properties = cls(
            json_data["name"],
            json_data["category"],
            properties_floating_type_to_python_type.get(
                json_data["elementClass"], json_data["elementClass"]
            ),
            json_data["dataFormat"],
            _extract_num_channels(
                json_data.get("num_channels"),
                dataset_path,
                json_data["name"],
                json_data["wkwResolutions"][0]
                if len(json_data["wkwResolutions"]) > 0
                else None,
            ),
            json_data["boundingBox"],
            default_view_configuration=json_data.get("defaultViewConfiguration"),
        )

        # add resolutions to LayerProperties
        for resolution in json_data["wkwResolutions"]:
            layer_properties._add_resolution(resolution_type._from_json(resolution))

        return layer_properties

    def _add_resolution(self, resolution: Resolution) -> None:
        self._wkw_magnifications.append(resolution)

    def _delete_resolution(self, resolution: str) -> None:
        resolutions_to_delete = [
            res for res in self._wkw_magnifications if res.mag == Mag(resolution)
        ]
        for res in resolutions_to_delete:
            self._wkw_magnifications.remove(res)

    def get_bounding_box(self) -> BoundingBox:

        return BoundingBox(self.get_bounding_box_offset(), self.get_bounding_box_size())

    def get_bounding_box_size(self) -> Tuple[int, int, int]:
        return (
            self.bounding_box["width"],
            self.bounding_box["height"],
            self.bounding_box["depth"],
        )

    def get_bounding_box_offset(self) -> Tuple[int, int, int]:
        return cast(Tuple[int, int, int], tuple(self.bounding_box["topLeft"]))

    def _set_bounding_box_size(self, size: Tuple[int, int, int]) -> None:
        # Cast to int in case the provided parameter contains numpy integer
        self._bounding_box["width"] = int(size[0])
        self._bounding_box["height"] = int(size[1])
        self._bounding_box["depth"] = int(size[2])

    def _set_bounding_box_offset(self, offset: Tuple[int, int, int]) -> None:
        # Cast to int in case the provided parameter contains numpy integer
        self._bounding_box["topLeft"] = cast(
            Tuple[int, int, int], tuple(map(int, offset))
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def category(self) -> str:
        return self._category

    @property
    def element_class(self) -> str:
        return self._element_class

    @property
    def data_format(self) -> str:
        return self._data_format

    @property
    def num_channels(self) -> int:
        return self._num_channels

    @property
    def bounding_box(self) -> dict:
        return self._bounding_box

    @property
    def wkw_magnifications(self) -> List[Resolution]:
        return self._wkw_magnifications

    @property
    def default_view_configuration(self) -> Optional[dict]:
        return self._default_view_configuration


class SegmentationLayerProperties(LayerProperties):
    def __init__(
        self,
        name: str,
        category: str,
        element_class: str,
        data_format: str,
        num_channels: int,
        bounding_box: Dict[str, Union[int, Tuple[int, int, int]]] = None,
        resolutions: List[Resolution] = None,
        largest_segment_id: int = None,
        mappings: List[str] = None,
        default_view_configuration: Optional[dict] = None,
    ) -> None:
        super().__init__(
            name,
            category,
            element_class,
            data_format,
            num_channels,
            bounding_box,
            resolutions,
            default_view_configuration,
        )
        # The parameter largest_segment_id is in fact not optional.
        # However, specifying the parameter as not optional, would require to change the parameter order
        assert largest_segment_id is not None
        self._largest_segment_id = largest_segment_id
        self._mappings = mappings

    def _to_json(self) -> Dict[str, Any]:
        json_properties = super()._to_json()
        json_properties["largestSegmentId"] = self._largest_segment_id
        if self._mappings is not None:
            json_properties["mappings"] = self._mappings
        return json_properties

    @classmethod
    def _from_json(
        cls,
        json_data: Dict[str, Any],
        resolution_type: Type[Resolution],
        dataset_path: Path,
    ) -> "SegmentationLayerProperties":
        if "largestSegmentId" not in json_data:
            warnings.warn(
                f"Segmentation layer {json_data['name']} is missing the 'largestSegmentId', defaulting to -1.",
                RuntimeWarning,
            )
        # create LayerProperties without resolutions
        layer_properties = cls(
            json_data["name"],
            json_data["category"],
            properties_floating_type_to_python_type.get(
                json_data["elementClass"], json_data["elementClass"]
            ),
            json_data["dataFormat"],
            _extract_num_channels(
                json_data.get("num_channels"),
                dataset_path,
                json_data["name"],
                json_data["wkwResolutions"][0]
                if len(json_data["wkwResolutions"]) > 0
                else None,
            ),
            json_data["boundingBox"],
            None,
            json_data.get("largestSegmentId", -1),
            json_data.get("mappings"),
            json_data.get("defaultViewConfiguration"),
        )

        # add resolutions to LayerProperties
        for resolution in json_data["wkwResolutions"]:
            layer_properties._add_resolution(resolution_type._from_json(resolution))

        return layer_properties

    @property
    def largest_segment_id(self) -> int:
        assert self._largest_segment_id is not None
        return self._largest_segment_id

    @property
    def mappings(self) -> Optional[List[str]]:
        return self._mappings
