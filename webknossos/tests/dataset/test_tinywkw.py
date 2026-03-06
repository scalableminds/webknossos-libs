"""Tests comparing tinywkw (pure-Python) against the native wkw module."""

import numpy as np
import pytest
import wkw
from upath import UPath

from tests.constants import TESTDATA_DIR
from webknossos.dataset._utils.tinywkw import WkwDataset, WkwHeader
from webknossos.geometry import NormalizedBoundingBox

DATASET_PATH = TESTDATA_DIR / "simple_wkw_dataset" / "color" / "1"

# block_len=8, file_len=4 → shard=32; num_channels=3, dtype=uint8
BLOCK_LEN = 8
FILE_LEN = 4
SHARD = FILE_LEN * BLOCK_LEN  # 32
NUM_CHANNELS = 3


def _make_bbox(x: int, y: int, z: int, sx: int, sy: int, sz: int) -> NormalizedBoundingBox:
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
    return ds.read(bbox)


# ---------------------------------------------------------------------------
# Header tests
# ---------------------------------------------------------------------------


def test_header_magic_and_version() -> None:
    raw = (DATASET_PATH / "header.wkw").read_bytes()
    header = WkwHeader.from_bytes(raw)
    # Verify the fields we know from introspecting the file
    assert header.block_len == BLOCK_LEN
    assert header.file_len == FILE_LEN
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
        (BLOCK_LEN, BLOCK_LEN, BLOCK_LEN),  # exactly one block
        (SHARD, SHARD, SHARD),  # full shard
        (5, 3, 7),  # sub-block, non-aligned
        (SHARD + BLOCK_LEN, SHARD, SHARD),  # spans two shards in x
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
        ((0, 0, 0), (BLOCK_LEN, BLOCK_LEN, BLOCK_LEN)),  # one block, aligned
        ((0, 0, 0), (SHARD, SHARD, SHARD)),  # full shard
        ((3, 5, 7), (5, 3, 1)),  # sub-block unaligned
        ((16, 16, 16), (SHARD, SHARD, SHARD)),  # cross-shard (missing file → zeros)
        ((0, 0, 0), (SHARD + BLOCK_LEN, SHARD, SHARD)),  # x spans two shards
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
    bbox = _make_bbox(SHARD, SHARD, SHARD, BLOCK_LEN, BLOCK_LEN, BLOCK_LEN)
    data = _tiny_read(bbox)
    assert data.shape == (NUM_CHANNELS, BLOCK_LEN, BLOCK_LEN, BLOCK_LEN)
    assert not data.any()
