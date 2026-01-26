import warnings
from abc import abstractmethod
from collections.abc import Iterator
from os import PathLike
from os.path import relpath
from typing import TYPE_CHECKING

from upath import UPath

from webknossos.dataset.transfer_mode import TransferMode
from webknossos.dataset_properties import AttachmentDataFormat, AttachmentsProperties
from webknossos.utils import (
    cheap_resolve,
    copytree,
    enrich_path,
    is_fs_path,
    snake_to_camel_case,
    warn_deprecated,
)

from .attachment import (
    AgglomerateAttachment,
    Attachment,
    ConnectomeAttachment,
    CumsumAttachment,
    MeshAttachment,
    SegmentIndexAttachment,
)

if TYPE_CHECKING:
    from webknossos.dataset.layer import (
        RemoteSegmentationLayer,
        SegmentationLayer,
    )
    from webknossos.dataset.layer.segmentation_layer.abstract_segmentation_layer import (
        AbstractSegmentationLayer,
    )


def _maybe_add_suffix(attachment_name: str, data_format: AttachmentDataFormat) -> str:
    if data_format == AttachmentDataFormat.Zarr3:
        return attachment_name
    return f"{attachment_name}.{data_format.value.lower()}"


class AbstractAttachments:
    _layer: "AbstractSegmentationLayer"

    def __init__(self, layer: "AbstractSegmentationLayer"):
        self._layer = layer

    @property
    def meshes(self) -> tuple[MeshAttachment, ...]:
        if self._properties.meshes is None:
            return tuple()
        return tuple(
            MeshAttachment(
                attachment,
                enrich_path(attachment.path, self._get_optional_dataset_path()),
            )
            for attachment in self._properties.meshes
        )

    @property
    def agglomerates(self) -> tuple[AgglomerateAttachment, ...]:
        if self._properties.agglomerates is None:
            return tuple()
        return tuple(
            AgglomerateAttachment(
                attachment,
                enrich_path(attachment.path, self._get_optional_dataset_path()),
            )
            for attachment in self._properties.agglomerates
        )

    @property
    def segment_index(self) -> SegmentIndexAttachment | None:
        if self._properties.segment_index is None:
            return None
        return SegmentIndexAttachment(
            self._properties.segment_index,
            enrich_path(
                self._properties.segment_index.path,
                self._get_optional_dataset_path(),
            ),
        )

    @property
    def cumsum(self) -> CumsumAttachment | None:
        if self._properties.cumsum is None:
            return None
        return CumsumAttachment(
            self._properties.cumsum,
            enrich_path(
                self._properties.cumsum.path, self._get_optional_dataset_path()
            ),
        )

    @property
    def connectomes(self) -> tuple[ConnectomeAttachment, ...]:
        if self._properties.connectomes is None:
            return tuple()
        return tuple(
            ConnectomeAttachment(
                attachment,
                enrich_path(attachment.path, self._get_optional_dataset_path()),
            )
            for attachment in self._properties.connectomes
        )

    @abstractmethod
    def _get_optional_dataset_path(self) -> UPath | None:
        """
        Attachment paths can be relative to the dataset path.
        But attachments store absolute paths only.
        In case we have a dataset path we can make a relative attachment path absolute.
        But if we don't, we fail when encountering a relative attachment path.
        """
        pass

    @property
    def _properties(self) -> "AttachmentsProperties":
        return self._layer._properties.attachments

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
            setattr(self._properties, container_name, attachment._properties)
        else:
            if any(a.name == attachment.name for a in getattr(self, container_name)):
                raise ValueError(
                    f"Attachment with name {attachment.name} already exists in {container_name}."
                )
            properties_container = getattr(self._properties, container_name)
            if properties_container is None:
                setattr(self._properties, container_name, [attachment._properties])
            else:
                properties_container.append(attachment._properties)
        self._save_properties()

    def add_copy_attachments(self, *other: Attachment) -> list[Attachment]:
        warn_deprecated("add_copy_attachments", "add_attachment_as_copy")
        return [self.add_attachment_as_copy(*other) for attachment in other]

    @abstractmethod
    def add_attachment_as_copy(self, attachment: Attachment) -> Attachment:
        pass


class RemoteAttachments(AbstractAttachments):
    _layer: "RemoteSegmentationLayer"

    def __init__(self, layer: "RemoteSegmentationLayer"):
        super().__init__(layer)

    def _apply_server_properties(self) -> None:
        self._layer._apply_server_layer_properties()

    def _get_optional_dataset_path(self) -> UPath | None:
        return None

    def add_attachment_as_copy(self, attachment: Attachment) -> Attachment:
        return self.upload_attachment(attachment, transfer_mode=TransferMode.COPY)

    def upload_attachment(
        self,
        attachment: Attachment,
        transfer_mode: TransferMode = TransferMode.COPY,
        common_storage_prefix: str | None = None,
    ) -> Attachment:
        self._ensure_writable()
        if transfer_mode not in (
            TransferMode.COPY,
            TransferMode.MOVE_AND_SYMLINK,
            TransferMode.SYMLINK,
        ):
            raise ValueError(f"Transfer mode {transfer_mode} is not supported.")

        target_dataset_id = self._layer.dataset.dataset_id
        from webknossos.client.context import _get_api_client

        client = _get_api_client()
        new_path = enrich_path(
            client.reserve_attachment_upload_to_path(
                target_dataset_id,
                self._layer.name,
                attachment.name,
                attachment.type_name,
                str(attachment.data_format),
                common_storage_prefix,
            )
        )
        # transfer to target dataset
        transfer_mode.transfer(
            attachment.path, new_path, progress_desc_label="attachment"
        )

        client.finish_attachment_upload_to_path(
            target_dataset_id,
            self._layer.name,
            attachment.name,
            attachment.type_name,
            str(attachment.data_format),
            common_storage_prefix,
        )

        # sync to server state
        self._apply_server_properties()

        new_attachment = type(attachment).from_path_and_name(
            new_path,
            attachment.name,
            data_format=attachment.data_format,
        )

        return new_attachment


class Attachments(AbstractAttachments):
    _layer: "SegmentationLayer"

    def __init__(self, layer: "SegmentationLayer"):
        super().__init__(layer)

    def _get_optional_dataset_path(self) -> UPath:
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
            setattr(self._properties, container_name, None)
        else:
            properties_container = getattr(self._properties, container_name)
            properties_container.remove(attachment._properties)
            if len(properties_container) == 0:
                setattr(self._properties, container_name, None)
        self._save_properties()

    def add_mesh(
        self,
        path: str | PathLike | UPath,
        *,
        name: str,
        data_format: AttachmentDataFormat,
    ) -> MeshAttachment:
        attachment = MeshAttachment.from_path_and_name(
            UPath(path),
            name,
            data_format=data_format,
            dataset_path=self._get_optional_dataset_path(),
        )
        self._add_attachment(attachment)
        return attachment

    def add_agglomerate(
        self,
        path: str | PathLike | UPath,
        *,
        name: str,
        data_format: AttachmentDataFormat,
    ) -> AgglomerateAttachment:
        attachment = AgglomerateAttachment.from_path_and_name(
            UPath(path),
            name,
            data_format=data_format,
            dataset_path=self._get_optional_dataset_path(),
        )
        self._add_attachment(attachment)
        return attachment

    def add_connectome(
        self,
        path: str | PathLike | UPath,
        *,
        name: str,
        data_format: AttachmentDataFormat,
    ) -> ConnectomeAttachment:
        attachment = ConnectomeAttachment.from_path_and_name(
            UPath(path),
            name,
            data_format=data_format,
            dataset_path=self._get_optional_dataset_path(),
        )
        self._add_attachment(attachment)
        return attachment

    def set_segment_index(
        self,
        path: str | PathLike | UPath,
        *,
        name: str,
        data_format: AttachmentDataFormat,
    ) -> SegmentIndexAttachment:
        attachment = SegmentIndexAttachment.from_path_and_name(
            UPath(path),
            name,
            data_format=data_format,
            dataset_path=self._get_optional_dataset_path(),
        )
        self._add_attachment(attachment)
        return attachment

    def set_cumsum(
        self,
        path: str | PathLike | UPath,
        *,
        name: str,
        data_format: AttachmentDataFormat,
    ) -> CumsumAttachment:
        attachment = CumsumAttachment.from_path_and_name(
            UPath(path),
            name,
            data_format=data_format,
            dataset_path=self._get_optional_dataset_path(),
        )
        self._add_attachment(attachment)
        return attachment

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

    def add_attachments(self, *other: Attachment) -> list[Attachment]:
        warn_deprecated("add_attachments", "add_attachment_as_ref")
        return [self.add_attachment_as_ref(attachment) for attachment in other]

    def add_attachment_as_ref(self, attachment: Attachment) -> Attachment:
        new_attachment = type(attachment).from_path_and_name(
            cheap_resolve(attachment.path),
            attachment.name,
            data_format=attachment.data_format,
            dataset_path=self._get_optional_dataset_path(),
        )
        self._add_attachment(new_attachment)
        return new_attachment

    def add_attachment_as_copy(self, attachment: Attachment) -> Attachment:
        self._ensure_writable()
        # We set the attachment path following the convention.
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
        return new_attachment

    def add_symlink_attachments(
        self, *other: Attachment, make_relative: bool = False
    ) -> list[Attachment]:
        warnings.warn(
            "Using symlinks is deprecated and will be removed in a future version. "
            + "Use `add_attachment_as_ref` instead, which adds an attachment as a reference to the layer.",
            DeprecationWarning,
            stacklevel=2,
        )
        output = []
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
                    assert is_fs_path(new_path.parent), (
                        "Relative symlinks are only supported for local paths."
                    )
                    new_path.symlink_to(
                        UPath(relpath(attachment.path, new_path.parent))
                    )
                else:
                    new_path.symlink_to(attachment.path)
            new_attachment = type(attachment).from_path_and_name(
                new_path,
                attachment.name,
                data_format=attachment.data_format,
                dataset_path=self._layer.dataset.resolved_path,
            )
            self._add_attachment(new_attachment)
            output.append(new_attachment)
        return output

    def detect_legacy_attachments(self) -> None:
        """Detects and adds legacy attachments.
        Legacy attachments are attachments that were stored in the folder hierarchy of the layer without explicit metadata."""
        if not is_fs_path(self._layer.resolved_path):
            return

        def _detect_hdf5(typ: type[Attachment]) -> None:
            folder_name = snake_to_camel_case(typ.container_name)
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
