from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self, cast, get_args, get_type_hints

from upath import UPath

from ..utils import dump_path, enrich_path
from .data_format import AttachmentDataFormat
from .properties import AttachmentProperties, AttachmentsProperties

if TYPE_CHECKING:
    from .layer import SegmentationLayer


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
    path: Path
    data_format: AttachmentDataFormat

    def __init__(
        self,
        properties: AttachmentProperties,
        path: Path,
    ):
        _validate_data_format(self.__class__, properties.data_format)
        self._properties = properties
        self.path = path
        self.data_format = properties.data_format

    @classmethod
    def from_path(
        cls, path: Path, data_format: AttachmentDataFormat, dataset_path: Path
    ) -> Self:
        return cls(
            AttachmentProperties(data_format, dump_path(path, dataset_path)),
            path,
        )


class MeshAttachment(Attachment):
    data_format: Literal[AttachmentDataFormat.ZARR3, AttachmentDataFormat.HDF5]


class SegmentIndexAttachment(Attachment):
    data_format: Literal[AttachmentDataFormat.ZARR3, AttachmentDataFormat.HDF5]


class AgglomerateAttachment(Attachment):
    data_format: Literal[AttachmentDataFormat.ZARR3, AttachmentDataFormat.HDF5]


class CumsumAttachment(Attachment):
    data_format: Literal[AttachmentDataFormat.ZARR3, AttachmentDataFormat.JSON]


class ConnectomeAttachment(Attachment):
    data_format: Literal[AttachmentDataFormat.ZARR3, AttachmentDataFormat.HDF5]


class Attachments:
    _layer: "SegmentationLayer"
    _properties: AttachmentsProperties
    meshes: tuple[MeshAttachment, ...] = ()
    agglomerates: tuple[AgglomerateAttachment, ...] = ()
    segment_index: SegmentIndexAttachment | None = None
    cumsum: CumsumAttachment | None = None
    connectomes: tuple[ConnectomeAttachment, ...] = ()

    def __init__(self, layer: "SegmentationLayer", properties: "AttachmentsProperties"):
        self._layer = layer
        self._properties = properties

        if properties.meshes is not None:
            self.meshes = tuple(
                MeshAttachment(
                    attachment,
                    enrich_path(attachment.path, layer.dataset.resolved_path),
                )
                for attachment in properties.meshes
            )
        if properties.agglomerates is not None:
            self.agglomerates = tuple(
                AgglomerateAttachment(
                    attachment,
                    enrich_path(attachment.path, layer.dataset.resolved_path),
                )
                for attachment in properties.agglomerates
            )
        if properties.segment_index is not None:
            self.segment_index = SegmentIndexAttachment(
                properties.segment_index,
                enrich_path(properties.segment_index.path, layer.dataset.resolved_path),
            )
        if properties.cumsum is not None:
            self.cumsum = CumsumAttachment(
                properties.cumsum,
                enrich_path(properties.cumsum.path, layer.dataset.resolved_path),
            )
        if properties.connectomes is not None:
            self.connectomes = tuple(
                ConnectomeAttachment(
                    attachment,
                    enrich_path(attachment.path, layer.dataset.resolved_path),
                )
                for attachment in properties.connectomes
            )

    def _add_attachment(
        self,
        attachment: Attachment,
        container_name: str,
    ) -> None:
        self._layer._ensure_writable()
        setattr(self, container_name, getattr(self, container_name) + (attachment,))
        properties_container = getattr(self._properties, container_name)
        if properties_container is None:
            setattr(self._properties, container_name, [attachment._properties])
        else:
            properties_container.append(attachment._properties)
        self._layer.dataset._export_as_json()

    def _set_attachment(
        self,
        attachment: Attachment,
        container_name: str,
    ) -> None:
        self._layer._ensure_writable()
        setattr(self, container_name, attachment)
        setattr(self._properties, container_name, attachment._properties)
        self._layer.dataset._export_as_json()

    def _remove_attachment(
        self,
        attachment: Attachment,
        container_name: str,
    ) -> None:
        setattr(
            self,
            container_name,
            tuple(p for p in getattr(self, container_name) if p != attachment),
        )
        properties_container = getattr(self._properties, container_name)
        properties_container.remove(attachment._properties)
        if len(properties_container) == 0:
            setattr(self._properties, container_name, None)
        self._layer.dataset._export_as_json()

    def _unset_attachment(
        self,
        attachment: Attachment,
        container_name: str,
    ) -> None:
        if getattr(self, container_name) != attachment:
            raise KeyError(f"Attachment {attachment} is not part of {container_name}.")
        setattr(self, container_name, None)
        setattr(self._properties, container_name, None)
        self._layer.dataset._export_as_json()

    def add_mesh(self, path: str | PathLike, data_format: AttachmentDataFormat) -> None:
        self._add_attachment(
            MeshAttachment.from_path(
                UPath(path), data_format, self._layer.dataset.resolved_path
            ),
            "meshes",
        )

    def add_agglomerate(
        self, path: str | PathLike, data_format: AttachmentDataFormat
    ) -> None:
        self._add_attachment(
            AgglomerateAttachment.from_path(
                UPath(path), data_format, self._layer.dataset.resolved_path
            ),
            "agglomerates",
        )

    def add_connectome(
        self, path: str | PathLike, data_format: AttachmentDataFormat
    ) -> None:
        self._add_attachment(
            ConnectomeAttachment.from_path(
                UPath(path), data_format, self._layer.dataset.resolved_path
            ),
            "connectomes",
        )

    def set_segment_index(
        self, path: str | PathLike, data_format: AttachmentDataFormat
    ) -> None:
        self._set_attachment(
            SegmentIndexAttachment.from_path(
                UPath(path), data_format, self._layer.dataset.resolved_path
            ),
            "segment_index",
        )

    def set_cumsum(
        self, path: str | PathLike, data_format: AttachmentDataFormat
    ) -> None:
        self._set_attachment(
            CumsumAttachment.from_path(
                UPath(path), data_format, self._layer.dataset.resolved_path
            ),
            "cumsum",
        )

    def delete_attachment(self, attachment: Attachment) -> None:
        self._layer._ensure_writable()
        if isinstance(attachment, MeshAttachment):
            self._remove_attachment(attachment, "meshes")
        elif isinstance(attachment, AgglomerateAttachment):
            self._remove_attachment(attachment, "agglomerates")
        elif isinstance(attachment, ConnectomeAttachment):
            self._remove_attachment(attachment, "connectomes")
        elif isinstance(attachment, SegmentIndexAttachment):
            self._unset_attachment(attachment, "segment_index")
        elif isinstance(attachment, CumsumAttachment):
            self._unset_attachment(attachment, "cumsum")
        else:
            raise TypeError(f"Cannot delete attachment of type {attachment.__class__}")

    @property
    def is_empty(self) -> bool:
        return (
            (len(self.meshes) == 0)
            and (len(self.agglomerates) == 0)
            and (len(self.connectomes) == 0)
            and (self.segment_index is None)
            and (self.cumsum is None)
        )
