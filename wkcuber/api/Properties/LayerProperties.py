from os.path import join, dirname, isfile

from wkw import wkw

from wkcuber.mag import Mag
from wkcuber.api.bounding_box import BoundingBox


def extract_num_channels(num_channels_in_properties, path, layer, mag):
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


class LayerProperties:
    def __init__(
        self,
        name,
        category,
        element_class,
        data_format,
        num_channels,
        bounding_box=None,
        resolutions=None,
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
        self._wkw_magnifications = resolutions or []

    def _to_json(self):
        return {
            "name": self.name,
            "category": self.category,
            "elementClass": self.element_class,
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

    @classmethod
    def _from_json(cls, json_data, resolution_type, dataset_path):
        # create LayerProperties without resolutions
        layer_properties = cls(
            json_data["name"],
            json_data["category"],
            json_data["elementClass"],
            json_data["dataFormat"],
            extract_num_channels(
                json_data.get("num_channels"),
                dataset_path,
                json_data["name"],
                json_data["wkwResolutions"][0]
                if len(json_data["wkwResolutions"]) > 0
                else None,
            ),
            json_data["boundingBox"],
        )

        # add resolutions to LayerProperties
        for resolution in json_data["wkwResolutions"]:
            layer_properties._add_resolution(resolution_type._from_json(resolution))

        return layer_properties

    def _add_resolution(self, resolution):
        self._wkw_magnifications.append(resolution)

    def _delete_resolution(self, resolution):
        resolutions_to_delete = [
            res for res in self._wkw_magnifications if res.mag == Mag(resolution)
        ]
        for res in resolutions_to_delete:
            self._wkw_magnifications.remove(res)

    def get_bounding_box(self) -> BoundingBox:

        return BoundingBox(self.get_bounding_box_offset(), self.get_bounding_box_size())

    def get_bounding_box_size(self) -> tuple:
        return (
            self.bounding_box["width"],
            self.bounding_box["height"],
            self.bounding_box["depth"],
        )

    def get_bounding_box_offset(self) -> tuple:
        return tuple(self.bounding_box["topLeft"])

    def _set_bounding_box_size(self, size):
        # Cast to int in case the provided parameter contains numpy integer
        self._bounding_box["width"] = int(size[0])
        self._bounding_box["height"] = int(size[1])
        self._bounding_box["depth"] = int(size[2])

    def _set_bounding_box_offset(self, offset):
        # Cast to int in case the provided parameter contains numpy integer
        self._bounding_box["topLeft"] = tuple(map(int, offset))

    @property
    def name(self) -> str:
        return self._name

    @property
    def category(self) -> str:
        return self._category

    @property
    def element_class(self):
        return self._element_class

    @property
    def data_format(self):
        return self._data_format

    @property
    def num_channels(self) -> int:
        return self._num_channels

    @property
    def bounding_box(self) -> dict:
        return self._bounding_box

    @property
    def wkw_magnifications(self) -> dict:
        return self._wkw_magnifications


class SegmentationLayerProperties(LayerProperties):
    def __init__(
        self,
        name,
        category,
        element_class,
        data_format,
        num_channels,
        bounding_box=None,
        resolutions=None,
        largest_segment_id=None,
        mappings=None,
    ):
        super().__init__(
            name,
            category,
            element_class,
            data_format,
            num_channels,
            bounding_box,
            resolutions,
        )
        self._largest_segment_id = largest_segment_id
        self._mappings = mappings

    def _to_json(self):
        json_properties = super()._to_json()
        json_properties["largestSegmentId"] = self._largest_segment_id
        if self._mappings is not None:
            json_properties["mappings"] = self._mappings
        return json_properties

    @classmethod
    def _from_json(cls, json_data, resolution_type, dataset_path):
        # create LayerProperties without resolutions
        layer_properties = cls(
            json_data["name"],
            json_data["category"],
            json_data["elementClass"],
            json_data["dataFormat"],
            extract_num_channels(
                json_data.get("num_channels"),
                dataset_path,
                json_data["name"],
                json_data["wkwResolutions"][0],
            ),
            json_data["boundingBox"],
            None,
            json_data["largestSegmentId"],
            json_data["mappings"] if "mappings" in json_data else None,
        )

        # add resolutions to LayerProperties
        for resolution in json_data["wkwResolutions"]:
            layer_properties._add_resolution(resolution_type._from_json(resolution))

        return layer_properties

    @property
    def largest_segment_id(self) -> int:
        return self._largest_segment_id

    @property
    def mappings(self) -> list:
        return self._mappings
