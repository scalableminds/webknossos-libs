"""Tests comparing tinywkw (pure-Python) against the native wkw module."""

import numpy as np
import pytest
import wkw
from upath import UPath

from tests.constants import TESTDATA_DIR
from webknossos.dataset._utils.tinywkw import WkwDataset, WkwHeader
from webknossos.geometry import NormalizedBoundingBox, Vec3Int

DATASET_PATH = TESTDATA_DIR / "simple_wkw_dataset" / "color" / "1"

# chunk_shape=Vec3Int(8,8,8), shard_shape=Vec3Int(4,4,4) → shard=32; num_channels=3, dtype=uint8
CHUNK_LEN = 8
SHARD_LEN = 4
SHARD = SHARD_LEN * CHUNK_LEN  # 32
NUM_CHANNELS = 3


def _make_bbox(
    x: int, y: int, z: int, sx: int, sy: int, sz: int
) -> NormalizedBoundingBox:
    return NormalizedBoundingBox(
        topleft=(0, x, y, z),
        size=(NUM_CHANNELS, sx, sy, sz),
        axes=("c", "x", "y", "z"),
    )


def _wkw_read(offset: tuple[int, int, int], size: tuple[int, int, int]) -> np.ndarray:
    with wkw.Dataset.open(str(DATASET_PATH)) as ds:
        return ds.read(offset, size)


def _tiny_read(bbox: NormalizedBoundingBox) -> np.ndarray:
    ds = WkwDataset.open(UPath(DATASET_PATH))
    return ds.read_bbox(bbox)


# ---------------------------------------------------------------------------
# Header tests
# ---------------------------------------------------------------------------


def test_header_magic_and_version() -> None:
    raw = (DATASET_PATH / "header.wkw").read_bytes()
    header = WkwHeader.from_bytes(raw)
    # Verify the fields we know from introspecting the file
    assert header.chunk_shape == Vec3Int.full(CHUNK_LEN)
    assert header.shard_shape == Vec3Int.full(SHARD_LEN) * Vec3Int.full(CHUNK_LEN)
    assert header.num_channels == NUM_CHANNELS
    assert header.voxel_dtype == np.dtype("uint8")
    # header.wkw stores data_offset=0 (no actual data); shard files store 16 for RAW
    assert header.data_offset == 0


def test_header_invalid_magic_raises() -> None:
    raw = bytearray((DATASET_PATH / "header.wkw").read_bytes())
    raw[0:3] = b"BAD"
    with pytest.raises(AssertionError):
        WkwHeader.from_bytes(bytes(raw))


def test_header_invalid_version_raises() -> None:
    raw = bytearray((DATASET_PATH / "header.wkw").read_bytes())
    raw[3] = 2  # wrong version
    with pytest.raises(AssertionError):
        WkwHeader.from_bytes(bytes(raw))


# ---------------------------------------------------------------------------
# Shape / dtype tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "size",
    [
        (CHUNK_LEN, CHUNK_LEN, CHUNK_LEN),  # exactly one chunk
        (SHARD, SHARD, SHARD),  # full shard
        (5, 3, 7),  # sub-chunk, non-aligned
        (SHARD + CHUNK_LEN, SHARD, SHARD),  # spans two shards in x
    ],
)
def test_output_shape_and_dtype(size: tuple[int, int, int]) -> None:
    bbox = _make_bbox(0, 0, 0, *size)
    data = _tiny_read(bbox)
    assert data.shape == (NUM_CHANNELS, *size)
    assert data.dtype == np.dtype("uint8")


# ---------------------------------------------------------------------------
# Agreement with native wkw module
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "offset,size",
    [
        ((0, 0, 0), (CHUNK_LEN, CHUNK_LEN, CHUNK_LEN)),  # one chunk, aligned
        ((0, 0, 0), (SHARD, SHARD, SHARD)),  # full shard
        ((3, 5, 7), (5, 3, 1)),  # sub-chunk unaligned
        ((16, 16, 16), (SHARD, SHARD, SHARD)),  # cross-shard (missing file → zeros)
        ((0, 0, 0), (SHARD + CHUNK_LEN, SHARD, SHARD)),  # x spans two shards
    ],
)
def test_matches_native_wkw(
    offset: tuple[int, int, int], size: tuple[int, int, int]
) -> None:
    expected = _wkw_read(offset, size)
    bbox = _make_bbox(*offset, *size)
    actual = _tiny_read(bbox)
    np.testing.assert_array_equal(actual, expected)


# ---------------------------------------------------------------------------
# Missing-file zero-fill
# ---------------------------------------------------------------------------


def test_missing_shard_returns_zeros() -> None:
    """Reading a region whose shard file doesn't exist should return all zeros."""
    # Only z0/y0/x0.wkw exists; a non-zero shard index file is absent.
    bbox = _make_bbox(SHARD, SHARD, SHARD, CHUNK_LEN, CHUNK_LEN, CHUNK_LEN)
    data = _tiny_read(bbox)
    assert data.shape == (NUM_CHANNELS, CHUNK_LEN, CHUNK_LEN, CHUNK_LEN)
    assert not data.any()


# ---------------------------------------------------------------------------
# Write tests
# ---------------------------------------------------------------------------


@pytest.fixture
def fresh_dataset(tmp_path: UPath) -> tuple[WkwDataset, UPath]:
    (tmp_path / "header.wkw").write_bytes((DATASET_PATH / "header.wkw").read_bytes())
    return WkwDataset.open(UPath(tmp_path)), UPath(tmp_path)


def test_write_full_shard_roundtrip(fresh_dataset: tuple[WkwDataset, UPath]) -> None:
    from unittest.mock import patch

    ds, _ = fresh_dataset
    rng = np.random.default_rng(42)
    data = rng.integers(0, 256, (NUM_CHANNELS, SHARD, SHARD, SHARD), dtype=np.uint8)
    with patch.object(ds, "_read_shard_into", wraps=ds._read_shard_into) as mock_read:
        ds.write(Vec3Int(0, 0, 0), data)
        mock_read.assert_not_called()
    result = ds.read(Vec3Int(0, 0, 0), Vec3Int(SHARD, SHARD, SHARD))
    np.testing.assert_array_equal(result, data)


def test_write_partial_chunk_roundtrip(fresh_dataset: tuple[WkwDataset, UPath]) -> None:
    ds, _ = fresh_dataset
    rng = np.random.default_rng(7)
    data = rng.integers(0, 256, (NUM_CHANNELS, 5, 3, 7), dtype=np.uint8)
    ds.write(Vec3Int(3, 5, 1), data)
    result = ds.read(Vec3Int(3, 5, 1), Vec3Int(5, 3, 7))
    np.testing.assert_array_equal(result, data)


def test_write_preserves_unwritten_region(
    fresh_dataset: tuple[WkwDataset, UPath],
) -> None:
    ds, _ = fresh_dataset
    rng = np.random.default_rng(13)
    first = rng.integers(0, 256, (NUM_CHANNELS, SHARD, SHARD, SHARD), dtype=np.uint8)
    ds.write(Vec3Int(0, 0, 0), first)
    # Overwrite only a sub-region
    patch = rng.integers(
        0, 256, (NUM_CHANNELS, CHUNK_LEN, CHUNK_LEN, CHUNK_LEN), dtype=np.uint8
    )
    ds.write(Vec3Int(0, 0, 0), patch)
    result = ds.read(Vec3Int(0, 0, 0), Vec3Int(SHARD, SHARD, SHARD))
    np.testing.assert_array_equal(result[:, :CHUNK_LEN, :CHUNK_LEN, :CHUNK_LEN], patch)
    np.testing.assert_array_equal(
        result[:, CHUNK_LEN:, :, :], first[:, CHUNK_LEN:, :, :]
    )


def test_write_cross_shard(fresh_dataset: tuple[WkwDataset, UPath]) -> None:
    ds, _ = fresh_dataset
    rng = np.random.default_rng(99)
    # Region that straddles the boundary between shard x=0 and shard x=1
    offset = Vec3Int(SHARD - CHUNK_LEN // 2, 0, 0)
    size = Vec3Int(CHUNK_LEN, SHARD, SHARD)
    data = rng.integers(0, 256, (NUM_CHANNELS,) + size.to_tuple(), dtype=np.uint8)
    ds.write(offset, data)
    result = ds.read(offset, size)
    np.testing.assert_array_equal(result, data)


def test_write_readable_by_native_wkw(fresh_dataset: tuple[WkwDataset, UPath]) -> None:
    ds, path = fresh_dataset
    rng = np.random.default_rng(42)
    data = rng.integers(0, 256, (NUM_CHANNELS, SHARD, SHARD, SHARD), dtype=np.uint8)
    ds.write(Vec3Int(0, 0, 0), data)
    with wkw.Dataset.open(str(path)) as native:
        result = native.read((0, 0, 0), (SHARD, SHARD, SHARD))
    np.testing.assert_array_equal(result, data)
