from typing import Any

import numpy as np
import pytest
from upath import UPath

from tests.utils import write_n5_array
from webknossos.dataset import CorruptImageError
from webknossos.dataset._image_conversion.image_source import ReadOptions
from webknossos.dataset._image_conversion.n5_image_source import N5ImageSource


def _open(path: UPath, **options: Any) -> N5ImageSource:
    return N5ImageSource(path, ReadOptions(**options))


def test_plain_n5_dataset(tmp_upath: UPath) -> None:
    path = tmp_upath / "plain.n5"
    data = np.arange(3 * 5 * 7, dtype=np.uint16).reshape(3, 5, 7)
    write_n5_array(path, data)

    source = _open(path)

    assert source.dtype == np.dtype("uint16")
    assert source.num_channels == 1
    assert source.get_possible_layers() is None
    assert (source._z, source._y, source._x) == (3, 5, 7)

    block = source._read_source_box(
        timepoint=0, z=slice(0, 3), y=slice(0, 5), x=slice(0, 7)
    )
    np.testing.assert_array_equal(block[0], data)


def test_n5_pyramid_picks_finest_level(tmp_upath: UPath) -> None:
    group_path = tmp_upath / "pyramid.n5"
    finest = np.arange(4 * 8 * 8, dtype=np.uint8).reshape(4, 8, 8)
    coarse = np.zeros((4, 4, 4), dtype=np.uint8)
    write_n5_array(group_path / "s0", finest, downsampling_factors=[1, 1, 1])
    write_n5_array(group_path / "s1", coarse, downsampling_factors=[1, 2, 2])

    source = _open(group_path)

    assert (source._z, source._y, source._x) == (4, 8, 8)
    assert source.get_possible_layers() == {"scale": [0, 1]}
    block = source._read_source_box(
        timepoint=0, z=slice(0, 4), y=slice(0, 8), x=slice(0, 8)
    )
    np.testing.assert_array_equal(block[0], finest)


def test_n5_pyramid_scale_option_picks_other_level(tmp_upath: UPath) -> None:
    group_path = tmp_upath / "pyramid_pick.n5"
    finest = np.ones((4, 8, 8), dtype=np.uint8)
    coarse = np.full((4, 4, 4), 7, dtype=np.uint8)
    write_n5_array(group_path / "s0", finest)
    write_n5_array(group_path / "s1", coarse)

    source = _open(group_path, format_options={"scale": 1})

    assert (source._z, source._y, source._x) == (4, 4, 4)
    block = source._read_source_box(
        timepoint=0, z=slice(0, 4), y=slice(0, 4), x=slice(0, 4)
    )
    np.testing.assert_array_equal(block[0], coarse)


def test_n5_pyramid_without_root_attributes(tmp_upath: UPath) -> None:
    # A pyramid group whose own root has no attributes.json — only its s0/s1
    # subdirectories do.
    group_path = tmp_upath / "no_root_attrs.n5"
    data = np.ones((2, 2, 2), dtype=np.uint8)
    write_n5_array(group_path / "s0", data)

    assert N5ImageSource.probe_directory(group_path)
    source = _open(group_path)
    assert (source._z, source._y, source._x) == (2, 2, 2)


def test_n5_pyramid_with_non_minimal_s0_raises(tmp_upath: UPath) -> None:
    # s0/s1 naming says s0 is finest, but downsamplingFactors disagrees.
    group_path = tmp_upath / "inconsistent.n5"
    data = np.ones((2, 2, 2), dtype=np.uint8)
    write_n5_array(group_path / "s0", data, downsampling_factors=[2, 2, 2])
    write_n5_array(group_path / "s1", data, downsampling_factors=[1, 1, 1])

    with pytest.raises(CorruptImageError):
        _open(group_path)


def test_not_a_valid_n5_store_raises_corrupt_image_error(tmp_upath: UPath) -> None:
    empty_dir = tmp_upath / "empty.n5"
    empty_dir.mkdir(parents=True)
    with pytest.raises(CorruptImageError):
        _open(empty_dir)
