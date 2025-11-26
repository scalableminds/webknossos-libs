from typing import Literal, cast, get_args, get_type_hints

from typing_extensions import Self
from upath import UPath

from webknossos.dataset_properties import AttachmentDataFormat, AttachmentProperties
from webknossos.utils import dump_path


def _validate_data_format(
    cls: type["Attachment"], data_format: AttachmentDataFormat
) -> None:
    hints = get_type_hints(cls)
    allowed_formats = cast(
        tuple[AttachmentDataFormat, ...], get_args(hints.get("data_format", ()))
    )
    if not allowed_formats:
        raise TypeError(f"No `data_format` annotation found for {cls.__name__}.")
    if data_format not in allowed_formats:
        raise ValueError(
            f"{data_format} is not a valid data_format for {cls.__name__}. Allowed: {allowed_formats}"
        )


class Attachment:
    _properties: AttachmentProperties
    name: str
    path: UPath
    data_format: AttachmentDataFormat
    container_name: Literal[
        "agglomerates", "meshes", "segment_index", "cumsum", "connectomes"
    ]
    """
    The container names are also used to derive the folder names to put the attachments.
    The container names are converted to camelCase to get the folder names.
    """
    type_name: Literal["mesh", "agglomerate", "segmentIndex", "connectome", "cumsum"]
    """
    The type names are used to communicate to WEBKNOSSOS which attachment type we want to e.g. upload.
    """

    def __init__(
        self,
        properties: AttachmentProperties,
        path: UPath,
    ):
        from webknossos.dataset.dataset import _assert_valid_layer_name

        _validate_data_format(self.__class__, properties.data_format)
        _assert_valid_layer_name(properties.name)
        self._properties = properties
        self.name = properties.name
        self.path = path
        self.data_format = properties.data_format

    @classmethod
    def from_path_and_name(
        cls,
        path: UPath,
        name: str,
        *,
        data_format: AttachmentDataFormat,
        dataset_path: UPath | None = None,
    ) -> Self:
        if not path.is_absolute():
            if dataset_path is None:
                raise ValueError(
                    "dataset_path must be provided when path is not absolute."
                )
            path = dataset_path / path
        return cls(
            AttachmentProperties(
                name=name, data_format=data_format, path=dump_path(path, dataset_path)
            ),
            path,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Attachment):
            return False
        return (
            self.name == other.name
            and self.path == other.path
            and self.data_format == other.data_format
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(path={repr(self.path)}, name={self.name}, data_format={self.data_format})"


class MeshAttachment(Attachment):
    data_format: Literal[AttachmentDataFormat.Zarr3, AttachmentDataFormat.HDF5]
    container_name = "meshes"
    type_name = "mesh"


class SegmentIndexAttachment(Attachment):
    data_format: Literal[AttachmentDataFormat.Zarr3, AttachmentDataFormat.HDF5]
    container_name = "segment_index"
    type_name = "segmentIndex"


class AgglomerateAttachment(Attachment):
    data_format: Literal[AttachmentDataFormat.Zarr3, AttachmentDataFormat.HDF5]
    container_name = "agglomerates"
    type_name = "agglomerate"


class CumsumAttachment(Attachment):
    data_format: Literal[AttachmentDataFormat.Zarr3, AttachmentDataFormat.JSON]
    container_name = "cumsum"
    type_name = "cumsum"


class ConnectomeAttachment(Attachment):
    data_format: Literal[AttachmentDataFormat.Zarr3, AttachmentDataFormat.HDF5]
    container_name = "connectomes"
    type_name = "connectome"
