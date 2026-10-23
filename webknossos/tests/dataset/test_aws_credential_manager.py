from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import tensorstore as ts
from upath import UPath

from webknossos.dataset._utils.tensorstore_helpers import AWSCredentialManager


def _open_s3_kvstore(aws_credentials: dict[str, str]) -> None:
    # Opening creates the credentials provider, which reads the credentials file;
    # nothing is requested from the endpoint.
    ts.KvStore.open(
        {
            "driver": "s3",
            "bucket": "test-bucket",
            "path": "data/",
            "endpoint": "http://127.0.0.1:9",
            "aws_credentials": aws_credentials,
        }
    ).result()


def test_concurrent_adds_keep_profiles_readable(tmp_path: Path) -> None:
    manager = AWSCredentialManager(UPath(tmp_path))

    def add_and_open(index: int) -> None:
        # Every 10th call registers new credentials, which rewrites the file
        key = f"key-{index // 10}" if index % 10 == 0 else "key-0"
        _open_s3_kvstore(manager.add(key, "secret"))

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(add_and_open, range(400)))


def test_adding_known_credentials_does_not_rewrite_the_file(tmp_path: Path) -> None:
    manager = AWSCredentialManager(UPath(tmp_path))
    first = manager.add("key", "secret")
    marker = "# not rewritten\n"
    with manager.credentials_file_path.open("a") as f:
        f.write(marker)

    assert manager.add("key", "secret") == first
    assert manager.credentials_file_path.read_text().endswith(marker)
    assert not list(tmp_path.glob("*.tmp"))


def test_failed_write_is_retried_on_next_add(tmp_path: Path) -> None:
    manager = AWSCredentialManager(UPath(tmp_path))
    original_dump = manager._dump_credentials

    def failing_dump() -> None:
        raise OSError("disk full")

    manager._dump_credentials = failing_dump  # type: ignore[method-assign]
    with pytest.raises(OSError):
        manager.add("key", "secret")
    assert manager.entries == {}

    manager._dump_credentials = original_dump  # type: ignore[method-assign]
    profile = manager.add("key", "secret")
    assert f"[{profile['profile']}]" in manager.credentials_file_path.read_text()
