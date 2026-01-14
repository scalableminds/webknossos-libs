import os
from enum import Enum

from upath import UPath

from webknossos.utils import copytree, is_fs_path


class TransferMode(Enum):
    """
    The transfer mode determines how mags or attachments are transferred to the remote or local storage.
    """

    HTTP = "http"
    MOVE_AND_SYMLINK = "move+symlink"
    SYMLINK = "symlink"
    COPY = "copy"

    @staticmethod
    def symlink(src_path: UPath, dst_path: UPath) -> None:
        assert is_fs_path(src_path) and is_fs_path(dst_path), (
            f"Either src_mag.path or mag.path are not pointing to a local file system {src_path}, {dst_path}"
        )
        assert not dst_path.exists(), f"Destination path {dst_path} already exists."
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        dst_path.symlink_to(src_path)

    @staticmethod
    def move_and_symlink(src_path: UPath, dst_path: UPath) -> None:
        assert is_fs_path(src_path) and is_fs_path(dst_path), (
            f"Either src_mag.path or mag.path are not pointing to a local file system {src_path}, {dst_path}"
        )
        assert not dst_path.exists(), f"Destination path {dst_path} already exists."
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        # We chose os.rename here to make sure we actually do a cheap filesystem operation.
        # We want to fail, if `rename` is not possible.
        os.rename(str(src_path), str(dst_path))
        src_path.symlink_to(dst_path.resolve())

    @staticmethod
    def copy(
        src_path: UPath, dst_path: UPath, progress_desc_label: str | None = None
    ) -> None:
        progress_desc_label = (
            progress_desc_label + " " if progress_desc_label is not None else ""
        )
        copytree(
            src_path,
            dst_path,
            progress_desc=f"copying {progress_desc_label}{src_path.path} to {dst_path.path}",
        )

    def transfer(
        self, src_path: UPath, dst_path: UPath, progress_desc_label: str | None = None
    ) -> None:
        if self.name == TransferMode.COPY.name:
            self.copy(src_path, dst_path, progress_desc_label)
        elif self.name == TransferMode.MOVE_AND_SYMLINK.name:
            self.move_and_symlink(src_path, dst_path)
        elif self.name == TransferMode.SYMLINK.name:
            self.symlink(src_path, dst_path)
        elif self.name == TransferMode.HTTP.name:
            raise NotImplementedError(
                "HTTP transfer mode is currently implemented via upload_dataset on the dataset level, not on individual mags or attachments."
            )
        else:
            raise ValueError(f"Not supported transfer mode {self.name}")
