import warnings
from abc import abstractmethod
from collections.abc import Iterator
from os import PathLike
from os.path import relpath
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast, get_args, get_type_hints

from typing_extensions import Self
from upath import UPath

from ..dataset_properties import (
    AttachmentDataFormat,
    AttachmentProperties,
    AttachmentsProperties,
)
from ..utils import (
    cheap_resolve,
    copytree,
    dump_path,
    enrich_path,
    is_fs_path,
    snake_to_camel_case,
    warn_deprecated,
)

if TYPE_CHECKING:
    from .layer import AbstractLayer, RemoteSegmentationLayer, SegmentationLayer


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


def _validate_name(name: str) -> None:
    from .dataset import _ALLOWED_LAYER_NAME_REGEX

    if not _ALLOWED_LAYER_NAME_REGEX.match(name):
        raise ValueError(
            f"Name {name} is not allowed. It must only contain letters, numbers, underscores, hyphens and dots."
        )


def _maybe_add_suffix(attachment_name: str, data_format: AttachmentDataFormat) -> str:
    if data_format == AttachmentDataFormat.Zarr3:
        return attachment_name
    return f"{attachment_name}.{data_format.value.lower()}"


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
        _validate_data_format(self.__class__, properties.data_format)
        _validate_name(properties.name)
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


class AbstractAttachments:
    _layer: "AbstractLayer"
    _properties: AttachmentsProperties
    meshes: tuple[MeshAttachment, ...] = ()
    agglomerates: tuple[AgglomerateAttachment, ...] = ()
    segment_index: SegmentIndexAttachment | None = None
    cumsum: CumsumAttachment | None = None
    connectomes: tuple[ConnectomeAttachment, ...] = ()

    def __init__(self, layer: "AbstractLayer", properties: "AttachmentsProperties"):
        self._properties = properties
        self._layer = layer
        optional_dataset_path = self._get_optional_dataset_path()

        if properties.meshes is not None:
            self.meshes = tuple(
                MeshAttachment(
                    attachment,
                    enrich_path(attachment.path, optional_dataset_path),
                )
                for attachment in properties.meshes
            )
        if properties.agglomerates is not None:
            self.agglomerates = tuple(
                AgglomerateAttachment(
                    attachment,
                    enrich_path(attachment.path, optional_dataset_path),
                )
                for attachment in properties.agglomerates
            )
        if properties.segment_index is not None:
            self.segment_index = SegmentIndexAttachment(
                properties.segment_index,
                enrich_path(properties.segment_index.path, optional_dataset_path),
            )
        if properties.cumsum is not None:
            self.cumsum = CumsumAttachment(
                properties.cumsum,
                enrich_path(properties.cumsum.path, optional_dataset_path),
            )
        if properties.connectomes is not None:
            self.connectomes = tuple(
                ConnectomeAttachment(
                    attachment,
                    enrich_path(attachment.path, optional_dataset_path),
                )
                for attachment in properties.connectomes
            )

    @abstractmethod
    def _get_optional_dataset_path(self) -> UPath | None:
        pass

    def _ensure_writable(self) -> None:
        self._layer._ensure_writable()

    def _save_properties(self) -> None:
        self._layer._save_layer_properties()

    @property
    def is_empty(self) -> bool:
        return (
            (len(self.meshes) == 0)
            and (len(self.agglomerates) == 0)
            and (len(self.connectomes) == 0)
            and (self.segment_index is None)
            and (self.cumsum is None)
        )

    def __iter__(self) -> Iterator[Attachment]:
        yield from (self.meshes or [])
        yield from (self.agglomerates or [])
        if self.segment_index is not None:
            yield self.segment_index
        if self.cumsum is not None:
            yield self.cumsum
        yield from (self.connectomes or [])

    def _add_attachment(
        self,
        attachment: Attachment,
    ) -> None:
        self._ensure_writable()
        container_name = attachment.container_name
        if isinstance(attachment, CumsumAttachment) or isinstance(
            attachment, SegmentIndexAttachment
        ):
            setattr(self, container_name, attachment)
            setattr(self._properties, container_name, attachment._properties)
        else:
            if any(a.name == attachment.name for a in getattr(self, container_name)):
                raise ValueError(
                    f"Attachment with name {attachment.name} already exists in {container_name}."
                )
            setattr(self, container_name, getattr(self, container_name) + (attachment,))
            properties_container = getattr(self._properties, container_name)
            if properties_container is None:
                setattr(self._properties, container_name, [attachment._properties])
            else:
                properties_container.append(attachment._properties)
        self._save_properties()

    def add_copy_attachments(self, *other: Attachment) -> None:
        warn_deprecated("add_copy_attachments", "add_attachment_as_copy")
        for attachment in other:
            self.add_attachment_as_copy(*other)

    @abstractmethod
    def add_attachment_as_copy(self, attachment: Attachment) -> None:
        pass


class RemoteAttachments(AbstractAttachments):
    _layer: "RemoteSegmentationLayer"

    def __init__(
        self, layer: "RemoteSegmentationLayer", properties: "AttachmentsProperties"
    ):
        super().__init__(layer, properties)

    def _get_optional_dataset_path(self) -> UPath | None:
        return None

    def add_attachment_as_copy(self, attachment: Attachment) -> None:
        self._ensure_writable()
        # In case of a remote dataset, we can ask wk for a path to put the attachment to.
        target_dataset_id = self._layer.dataset.dataset_id
        from webknossos.client.context import _get_context

        context = _get_context()
        new_path = enrich_path(
            context.api_client_with_auth.reserve_attachment_upload_to_path(
                target_dataset_id,
                self._layer.name,
                attachment.name,
                attachment.type_name,
                str(attachment.data_format),
            )
        )
        # copy to target dataset
        copytree(attachment.path, new_path)

        context.api_client_with_auth.finish_attachment_upload_to_path(
            target_dataset_id,
            self._layer.name,
            attachment.name,
            attachment.type_name,
            str(attachment.data_format),
        )
        new_attachment = type(attachment).from_path_and_name(
            new_path,
            attachment.name,
            data_format=attachment.data_format,
        )
        self._add_attachment(new_attachment)


class Attachments(AbstractAttachments):
    _layer: "SegmentationLayer"

    def __init__(self, layer: "SegmentationLayer", properties: "AttachmentsProperties"):
        super().__init__(layer, properties)

    def _get_optional_dataset_path(self) -> UPath | None:
        return self._layer.dataset.resolved_path

    def _remove_attachment(
        self,
        attachment: Attachment,
    ) -> None:
        self._ensure_writable()
        container_name = attachment.container_name
        if isinstance(attachment, CumsumAttachment) or isinstance(
            attachment, SegmentIndexAttachment
        ):
            if getattr(self, container_name) != attachment:
                raise KeyError(
                    f"Attachment {attachment} is not part of {container_name}."
                )
            setattr(self, container_name, None)
            setattr(self._properties, container_name, None)
        else:
            setattr(
                self,
                container_name,
                tuple(p for p in getattr(self, container_name) if p != attachment),
            )
            properties_container = getattr(self._properties, container_name)
            properties_container.remove(attachment._properties)
            if len(properties_container) == 0:
                setattr(self._properties, container_name, None)
        self._save_properties()

    def add_mesh(
        self, path: str | PathLike, *, name: str, data_format: AttachmentDataFormat
    ) -> None:
        self._add_attachment(
            MeshAttachment.from_path_and_name(
                UPath(path),
                name,
                data_format=data_format,
                dataset_path=self._get_optional_dataset_path(),
            )
        )

    def add_agglomerate(
        self, path: str | PathLike, *, name: str, data_format: AttachmentDataFormat
    ) -> None:
        self._add_attachment(
            AgglomerateAttachment.from_path_and_name(
                UPath(path),
                name,
                data_format=data_format,
                dataset_path=self._get_optional_dataset_path(),
            )
        )

    def add_connectome(
        self, path: str | PathLike, *, name: str, data_format: AttachmentDataFormat
    ) -> None:
        self._add_attachment(
            ConnectomeAttachment.from_path_and_name(
                UPath(path),
                name,
                data_format=data_format,
                dataset_path=self._get_optional_dataset_path(),
            )
        )

    def set_segment_index(
        self, path: str | PathLike, *, name: str, data_format: AttachmentDataFormat
    ) -> None:
        self._add_attachment(
            SegmentIndexAttachment.from_path_and_name(
                UPath(path),
                name,
                data_format=data_format,
                dataset_path=self._get_optional_dataset_path(),
            )
        )

    def set_cumsum(
        self, path: str | PathLike, *, name: str, data_format: AttachmentDataFormat
    ) -> None:
        self._add_attachment(
            CumsumAttachment.from_path_and_name(
                UPath(path),
                name,
                data_format=data_format,
                dataset_path=self._get_optional_dataset_path(),
            )
        )

    def delete_attachment(self, attachment: Attachment) -> None:
        if isinstance(attachment, MeshAttachment):
            self._remove_attachment(attachment)
        elif isinstance(attachment, AgglomerateAttachment):
            self._remove_attachment(attachment)
        elif isinstance(attachment, ConnectomeAttachment):
            self._remove_attachment(attachment)
        elif isinstance(attachment, SegmentIndexAttachment):
            self._remove_attachment(attachment)
        elif isinstance(attachment, CumsumAttachment):
            self._remove_attachment(attachment)
        else:
            raise TypeError(f"Cannot delete attachment of type {attachment.__class__}")

    def add_attachments(self, *other: Attachment) -> None:
        warn_deprecated("add_attachments", "add_attachment_as_ref")
        for attachment in other:
            self.add_attachment_as_ref(attachment)

    def add_attachment_as_ref(self, attachment: Attachment) -> None:
        new_attachment = type(attachment).from_path_and_name(
            cheap_resolve(attachment.path),
            attachment.name,
            data_format=attachment.data_format,
            dataset_path=self._get_optional_dataset_path(),
        )
        self._add_attachment(new_attachment)

    def add_attachment_as_copy(self, attachment: Attachment) -> None:
        self._ensure_writable()
        # In case we have a path, we put the attachment following the convention.
        new_path = cheap_resolve(
            self._layer.path
            / snake_to_camel_case(attachment.container_name)
            / _maybe_add_suffix(attachment.name, attachment.data_format)
        )
        new_path.parent.mkdir(parents=True, exist_ok=True)
        copytree(attachment.path, new_path)

        new_attachment = type(attachment).from_path_and_name(
            new_path,
            attachment.name,
            data_format=attachment.data_format,
            dataset_path=self._get_optional_dataset_path(),
        )

        self._add_attachment(new_attachment)

    def add_symlink_attachments(
        self, *other: Attachment, make_relative: bool = False
    ) -> None:
        warnings.warn(
            "Using symlinks is deprecated and will be removed in a future version. "
            + "Use `add_attachment_as_ref` instead, which adds an attachment as a reference to the layer.",
            DeprecationWarning,
            stacklevel=2,
        )
        for attachment in other:
            new_path = cheap_resolve(attachment.path)
            if is_fs_path(attachment.path):
                new_path = (
                    self._layer.resolved_path
                    / snake_to_camel_case(attachment.container_name)
                    / _maybe_add_suffix(attachment.name, attachment.data_format)
                )
                new_path.parent.mkdir(parents=True, exist_ok=True)
                if make_relative:
                    new_path.symlink_to(Path(relpath(attachment.path, new_path.parent)))
                else:
                    new_path.symlink_to(attachment.path)
            new_attachment = type(attachment).from_path_and_name(
                new_path,
                attachment.name,
                data_format=attachment.data_format,
                dataset_path=self._layer.dataset.resolved_path,
            )
            self._add_attachment(new_attachment)

    def detect_legacy_attachments(self) -> None:
        """Detects and adds legacy attachments.
        Legacy attachments are attachments that were stored in the folder hierarchy of the layer without explicit metadata."""
        if not is_fs_path(self._layer.resolved_path):
            return

        def _detect_hdf5(typ: type[Attachment]) -> None:
            folder_name = snake_to_camel_case(typ.container_name)
            if (
                self._layer.resolved_path is None
                or self._layer.dataset.resolved_path is None
            ):
                return
            if (self._layer.resolved_path / folder_name).exists():
                for attachment_path in (self._layer.resolved_path / folder_name).glob(
                    "*.hdf5"
                ):
                    new_attachment = typ.from_path_and_name(
                        attachment_path,
                        attachment_path.stem,
                        data_format=AttachmentDataFormat.HDF5,
                        dataset_path=self._layer.dataset.resolved_path,
                    )
                    self._add_attachment(new_attachment)

        _detect_hdf5(MeshAttachment)
        _detect_hdf5(AgglomerateAttachment)
        _detect_hdf5(ConnectomeAttachment)
        _detect_hdf5(SegmentIndexAttachment)

        if (self._layer.resolved_path / "agglomerates" / "cumsum.json").exists():
            self.set_cumsum(
                self._layer.resolved_path / "agglomerates" / "cumsum.json",
                name="cumsum",
                data_format=AttachmentDataFormat.JSON,
            )
