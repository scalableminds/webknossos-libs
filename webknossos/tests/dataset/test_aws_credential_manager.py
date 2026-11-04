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
        # Every 10th call registers new credentials, which writes a new file
        key = f"key-{index // 10}" if index % 10 == 0 else "key-0"
        _open_s3_kvstore(manager.add(key, "secret"))

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(add_and_open, range(400)))

    assert len(list(tmp_path.glob("credentials-*"))) == 40
    assert not list(tmp_path.glob("*.tmp"))


def test_adding_known_credentials_does_not_rewrite_the_file(tmp_path: Path) -> None:
    manager = AWSCredentialManager(UPath(tmp_path))
    first = manager.add("key", "secret")
    credentials_file = Path(first["credentials_file"])
    marker = "# not rewritten\n"
    with credentials_file.open("a") as f:
        f.write(marker)

    assert manager.add("key", "secret") == first
    assert credentials_file.read_text().endswith(marker)


def test_different_credentials_use_separate_files(tmp_path: Path) -> None:
    manager = AWSCredentialManager(UPath(tmp_path))
    first = manager.add("key", "secret")
    second = manager.add("key", "other-secret")

    assert first["profile"] != second["profile"]
    assert first["credentials_file"] != second["credentials_file"]
    assert "secret" not in first["credentials_file"]
    _open_s3_kvstore(first)
    _open_s3_kvstore(second)


def test_failed_write_is_retried_on_next_add(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = AWSCredentialManager(UPath(tmp_path))

    def failing_rename(*_args: object) -> None:
        raise OSError("disk full")

    with monkeypatch.context() as m:
        m.setattr(Path, "rename", failing_rename)
        with pytest.raises(OSError):
            manager.add("key", "secret")
    assert not list(tmp_path.glob("credentials-*"))
    assert not list(tmp_path.glob("*.tmp"))

    profile = manager.add("key", "secret")
    assert f"[{profile['profile']}]" in Path(profile["credentials_file"]).read_text()
