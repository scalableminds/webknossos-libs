from typing import Any

import numpy as np
import pytest
from upath import UPath

from tests.utils import write_neuroglancer_precomputed_scale
from webknossos.dataset import CorruptImageError
from webknossos.dataset._image_conversion.image_source import ReadOptions
from webknossos.dataset._image_conversion.neuroglancer_precomputed_image_source import (
    NeuroglancerPrecomputedImageSource,
)


def _open(path: UPath, **options: Any) -> NeuroglancerPrecomputedImageSource:
    return NeuroglancerPrecomputedImageSource(path, ReadOptions(**options))


def test_single_scale_volume(tmp_upath: UPath) -> None:
    path = tmp_upath / "volume"
    # (x, y, z, c) — the format's own native axis order.
    x, y, z = 7, 5, 3
    data = np.arange(x * y * z, dtype=np.uint8).reshape(x, y, z, 1)
    write_neuroglancer_precomputed_scale(path, data, resolution=(4.0, 4.0, 4.0))

    source = _open(path)

    assert source.dtype == np.dtype("uint8")
    assert source.num_channels == 1
    assert source.get_possible_layers() is None
    assert (source._x, source._y, source._z) == (x, y, z)

    block = source._read_source_box(
        timepoint=0, z=slice(0, z), y=slice(0, y), x=slice(0, x)
    )
    np.testing.assert_array_equal(
        block[0], data[:, :, :, 0].transpose(2, 1, 0)
    )  # (x, y, z) -> (z, y, x)


def test_multi_scale_volume_picks_finest(tmp_upath: UPath) -> None:
    path = tmp_upath / "pyramid"
    finest = np.arange(8 * 6 * 4, dtype=np.uint8).reshape(8, 6, 4, 1)
    coarse = np.zeros((4, 3, 2, 1), dtype=np.uint8)
    write_neuroglancer_precomputed_scale(path, finest, resolution=(4.0, 4.0, 4.0))
    write_neuroglancer_precomputed_scale(path, coarse, resolution=(8.0, 8.0, 8.0))

    source = _open(path)

    assert (source._x, source._y, source._z) == (8, 6, 4)
    assert source.get_possible_layers() == {"scale": [0, 1]}
    block = source._read_source_box(
        timepoint=0, z=slice(0, 4), y=slice(0, 6), x=slice(0, 8)
    )
    np.testing.assert_array_equal(block[0], finest[:, :, :, 0].transpose(2, 1, 0))


def test_multi_scale_volume_scale_option_picks_other_scale(tmp_upath: UPath) -> None:
    path = tmp_upath / "pyramid_pick"
    finest = np.ones((8, 6, 4, 1), dtype=np.uint8)
    coarse = np.full((4, 3, 2, 1), 7, dtype=np.uint8)
    write_neuroglancer_precomputed_scale(path, finest, resolution=(4.0, 4.0, 4.0))
    write_neuroglancer_precomputed_scale(path, coarse, resolution=(8.0, 8.0, 8.0))

    source = _open(path, format_options={"scale": 1})

    assert (source._x, source._y, source._z) == (4, 3, 2)
    block = source._read_source_box(
        timepoint=0, z=slice(0, 2), y=slice(0, 3), x=slice(0, 4)
    )
    np.testing.assert_array_equal(block[0], coarse[:, :, :, 0].transpose(2, 1, 0))


def test_multi_channel_volume(tmp_upath: UPath) -> None:
    path = tmp_upath / "multi_channel"
    x, y, z, c = 4, 3, 2, 3
    data = np.arange(x * y * z * c, dtype=np.uint8).reshape(x, y, z, c)
    write_neuroglancer_precomputed_scale(path, data)

    source = _open(path)

    # Three raw channels of a non-RGB-flagged format are offered as a split
    # (truncated to the first 3), same convention as every other reader.
    assert source.num_channels == 3
    assert source.get_possible_layers() == {"channel": [0, 1, 2]}


def test_probe_directory_requires_info_with_scales(tmp_upath: UPath) -> None:
    path = tmp_upath / "volume"
    data = np.ones((2, 2, 2, 1), dtype=np.uint8)
    write_neuroglancer_precomputed_scale(path, data)
    assert NeuroglancerPrecomputedImageSource.probe_directory(path)

    not_precomputed = tmp_upath / "not_precomputed"
    not_precomputed.mkdir(parents=True)
    assert not NeuroglancerPrecomputedImageSource.probe_directory(not_precomputed)


def test_missing_info_raises_corrupt_image_error(tmp_upath: UPath) -> None:
    empty_dir = tmp_upath / "empty"
    empty_dir.mkdir(parents=True)
    with pytest.raises(CorruptImageError):
        _open(empty_dir)


def test_info_without_scales_raises_corrupt_image_error(tmp_upath: UPath) -> None:
    path = tmp_upath / "no_scales"
    path.mkdir(parents=True)
    (path / "info").write_text(
        '{"@type": "neuroglancer_multiscale_volume", "data_type": "uint8", '
        '"num_channels": 1, "type": "image", "scales": []}'
    )
    with pytest.raises(CorruptImageError):
        _open(path)


def test_corrupt_info_raises_corrupt_image_error(tmp_upath: UPath) -> None:
    path = tmp_upath / "corrupt"
    path.mkdir(parents=True)
    (path / "info").write_text("{not valid json")
    with pytest.raises(CorruptImageError):
        _open(path)
