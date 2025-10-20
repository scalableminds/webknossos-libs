from enum import Enum
from shutil import move

from upath import UPath

from webknossos.utils import copytree, is_fs_path


class TransferMode(Enum):
    """
    The transfer mode determines how mags or attachments are transferred to the remote or local storage.
    """

    HTTP = "http"
    MOVE_AND_SYMLINK = "move+symlink"
    COPY = "copy"

    @staticmethod
    def move_and_symlink(src_path: UPath, dst_path: UPath) -> None:
        assert is_fs_path(src_path) and is_fs_path(dst_path), (
            f"Either src_mag.path or mag.path are not pointing to a local file system {src_path}, {dst_path}"
        )
        assert not dst_path.exists(), f"Destination path {dst_path} already exists."
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        move(str(src_path), str(dst_path))
        src_path.symlink_to(dst_path.resolve())

    @staticmethod
    def copy(src_path: UPath, dst_path: UPath) -> None:
        copytree(
            src_path,
            dst_path,
            progress_desc=f"copying attachment {src_path.path} to {dst_path.path}",
        )

    def transfer(self, src_path: UPath, dst_path: UPath) -> None:
        if self.name == TransferMode.COPY.name:
            self.copy(src_path, dst_path)
        elif self.name == TransferMode.MOVE_AND_SYMLINK.name:
            self.move_and_symlink(src_path, dst_path)
        elif self.name == TransferMode.HTTP.name:
            raise NotImplementedError(
                "HTTP transfer mode is currently implemented via upload_dataset on the dataset level, not on individual mags or attachments."
            )
        else:
            raise ValueError(f"Not supported transfer mode {self.name}")
