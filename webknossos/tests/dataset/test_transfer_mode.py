import tempfile

import pytest
from upath import UPath

from webknossos import TransferMode


def test_transfer_mode_move_and_symlink() -> None:
    transfer_mode = TransferMode.MOVE_AND_SYMLINK

    with tempfile.TemporaryDirectory() as tmp_dir:
        src = UPath(tmp_dir) / "src_test"
        dst = UPath(tmp_dir) / "dst_test"

        src.mkdir(parents=True, exist_ok=True)
        (src / "a.txt").touch()

        transfer_mode.transfer(src, dst)

        assert (dst / "a.txt").exists()
        assert (src / "a.txt").exists()
        assert src.is_symlink()

        # try to move again to same destination
        with pytest.raises(AssertionError):
            transfer_mode.transfer(src, dst)

        # try to move again to different destination
        dst2 = UPath(tmp_dir) / "dst_test2"
        transfer_mode.transfer(src, dst2)

        assert (dst / "a.txt").exists()
        assert not dst.is_symlink()
        assert (dst2 / "a.txt").exists()
        assert dst2.is_symlink()
        assert (src / "a.txt").exists()
        assert src.is_symlink()


def test_transfer_mode_symlink() -> None:
    transfer_mode = TransferMode.SYMLINK
    with tempfile.TemporaryDirectory() as tmp_dir:
        src = UPath(tmp_dir) / "src_test"
        dst = UPath(tmp_dir) / "dst_test"

        src.mkdir(parents=True, exist_ok=True)
        (src / "a.txt").touch()

        transfer_mode.transfer(src, dst)

        assert (dst / "a.txt").exists()
        assert (src / "a.txt").exists()
        assert dst.is_symlink()

        # try to move again to same destination
        with pytest.raises(AssertionError):
            transfer_mode.transfer(src, dst)

        # try to move again to different destination
        dst2 = UPath(tmp_dir) / "dst_test2"
        transfer_mode.transfer(src, dst2)

        assert (dst / "a.txt").exists()
        assert dst.is_symlink()
        assert (dst2 / "a.txt").exists()
        assert dst2.is_symlink()
        assert (src / "a.txt").exists()
        assert not src.is_symlink()
