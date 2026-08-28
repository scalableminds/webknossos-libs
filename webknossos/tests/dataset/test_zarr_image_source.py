import json
from typing import Any

import numpy as np
import pytest
from upath import UPath

from tests.data_fixtures import (
    write_ome_zarr_v2_group,
    write_ome_zarr_v3_group,
    write_zarr_v2_array,
    write_zarr_v3_array,
)
from webknossos.dataset import CorruptImageError, UnsupportedImageFormatError
from webknossos.dataset._image_conversion.image_source import ReadOptions
from webknossos.dataset._image_conversion.zarr_image_source import ZarrImageSource
from webknossos.geometry.constants import C_AXIS, CXYZ_AXES, X_AXIS, Y_AXIS, Z_AXIS


def _open(path: UPath, **options: Any) -> ZarrImageSource:
    return ZarrImageSource(path, ReadOptions(**options))


# Standard OME axes metadata (name/type), reused across the OME-Zarr tests.
_AXES_CZYX = [
    {"name": C_AXIS, "type": "channel"},
    {"name": Z_AXIS, "type": "space"},
    {"name": Y_AXIS, "type": "space"},
    {"name": X_AXIS, "type": "space"},
]


def test_plain_zarr_v2_array_uses_positional_axes(tmp_upath: UPath) -> None:
    # A bare .zarray carries no axis names, so a 3D array is assumed (z, y, x).
    array_path = tmp_upath / "plain.zarray_dir"
    data = np.arange(3 * 5 * 7, dtype=np.uint16).reshape(3, 5, 7)
    write_zarr_v2_array(array_path, data)

    source = _open(array_path)

    assert source.dtype == np.dtype("uint16")
    assert source.num_channels == 1
    assert source.get_layer_split_options() is None
    assert (source._z, source._y, source._x) == (3, 5, 7)

    block = source._read_source_box(
        timepoint=0, z=slice(0, 3), y=slice(0, 5), x=slice(0, 7)
    )
    np.testing.assert_array_equal(block[0], data)


def test_plain_zarr_v3_array_without_dimension_names_uses_positional_axes(
    tmp_upath: UPath,
) -> None:
    array_path = tmp_upath / "plain_v3"
    data = np.arange(3 * 5 * 7, dtype=np.uint8).reshape(3, 5, 7)
    write_zarr_v3_array(array_path, data)  # no dimension_names

    source = _open(array_path)

    assert (source._z, source._y, source._x) == (3, 5, 7)
    block = source._read_source_box(
        timepoint=0, z=slice(0, 3), y=slice(0, 5), x=slice(0, 7)
    )
    np.testing.assert_array_equal(block[0], data)


def test_plain_zarr_v3_array_with_dimension_names_uses_labels(tmp_upath: UPath) -> None:
    # dimension_names in a non-canonical physical order (c, x, y, z) proves
    # axis roles come from the labels, not from position.
    array_path = tmp_upath / "labeled_v3"
    c, x, y, z = 2, 7, 5, 3
    data = np.arange(c * x * y * z, dtype=np.uint8).reshape(c, x, y, z)
    write_zarr_v3_array(array_path, data, dimension_names=list(CXYZ_AXES))

    source = _open(array_path)

    assert (source._x, source._y, source._z) == (7, 5, 3)
    # Two raw channels are offered as a layer split (one pinned by default),
    # not written together — see compute_channel_selection.
    assert source.num_channels == 1
    assert source.get_layer_split_options() == {"channel": [0, 1]}

    block = source._read_source_box(
        timepoint=0, z=slice(0, z), y=slice(0, y), x=slice(0, x)
    )
    assert block.shape == (1, z, y, x)
    np.testing.assert_array_equal(
        block[0], data[0].transpose(2, 1, 0)
    )  # (x, y, z) -> (z, y, x), channel 0 (the default pin)


@pytest.mark.parametrize("writer", [write_ome_zarr_v2_group, write_ome_zarr_v3_group])
def test_ome_zarr_group_picks_finest_resolution(tmp_upath: UPath, writer: Any) -> None:
    group_path = tmp_upath / "multiscale"
    finest = np.arange(2 * 4 * 8 * 8, dtype=np.uint8).reshape(2, 4, 8, 8)
    coarse = np.zeros((2, 4, 4, 4), dtype=np.uint8)
    writer(
        group_path,
        [
            ("1", coarse, [1.0, 1.0, 2.0, 2.0]),  # listed first, but coarser
            ("0", finest, [1.0, 1.0, 1.0, 1.0]),
        ],
        _AXES_CZYX,
    )

    source = _open(group_path)

    assert (source._z, source._y, source._x) == (4, 8, 8)
    assert source.num_channels == 1
    # "scale" is never offered as a layer split — only "channel" is; the
    # resolution level always resolves to a single one, the finest by default.
    assert source.get_layer_split_options() == {"channel": [0, 1]}

    block = source._read_source_box(
        timepoint=0, z=slice(0, 4), y=slice(0, 8), x=slice(0, 8)
    )
    np.testing.assert_array_equal(block[0], finest[0])  # channel 0, the default pin


@pytest.mark.parametrize("writer", [write_ome_zarr_v2_group, write_ome_zarr_v3_group])
def test_ome_zarr_group_scale_option_picks_other_level(
    tmp_upath: UPath, writer: Any
) -> None:
    group_path = tmp_upath / "multiscale_pick"
    finest = np.ones((4, 8, 8), dtype=np.uint8)
    coarse = np.full((4, 4, 4), 7, dtype=np.uint8)
    axes = _AXES_CZYX[1:]  # z, y, x only — no channel axis in this group
    writer(
        group_path,
        [("0", finest, [1.0, 1.0, 1.0]), ("1", coarse, [1.0, 2.0, 2.0])],
        axes,
    )

    finest_source = _open(group_path)
    assert (finest_source._z, finest_source._y, finest_source._x) == (4, 8, 8)

    coarse_source = _open(group_path, format_options={"scale": 1})
    assert (coarse_source._z, coarse_source._y, coarse_source._x) == (4, 4, 4)
    block = coarse_source._read_source_box(
        timepoint=0, z=slice(0, 4), y=slice(0, 4), x=slice(0, 4)
    )
    np.testing.assert_array_equal(block[0], coarse)


def test_single_resolution_ome_zarr_group_reports_no_scale_choice(
    tmp_upath: UPath,
) -> None:
    group_path = tmp_upath / "single_scale"
    data = np.ones((4, 8, 8), dtype=np.uint8)
    write_ome_zarr_v2_group(group_path, [("0", data, [1.0, 1.0, 1.0])], _AXES_CZYX[1:])

    source = _open(group_path)
    assert source.get_layer_split_options() is None


def test_probe_directory_recognizes_bare_zarr_directories(tmp_upath: UPath) -> None:
    array_path = tmp_upath / "bare_array"  # no .zarr suffix
    write_zarr_v2_array(array_path, np.zeros((2, 2, 2), dtype=np.uint8))
    assert ZarrImageSource.probe_directory(array_path)

    group_path = tmp_upath / "bare_group"
    write_ome_zarr_v3_group(
        group_path,
        [("0", np.zeros((2, 2, 2), dtype=np.uint8), [1.0, 1.0, 1.0])],
        _AXES_CZYX[1:],
    )
    assert ZarrImageSource.probe_directory(group_path)

    assert not ZarrImageSource.probe_directory(tmp_upath / "not_a_store")


def test_remote_path_is_rejected() -> None:
    with pytest.raises(ValueError, match="local file path"):
        _open(UPath("memory://some/path.zarr"))


def test_not_a_zarr_store_raises_corrupt_image_error(tmp_upath: UPath) -> None:
    empty_dir = tmp_upath / "empty"
    empty_dir.mkdir(parents=True)
    with pytest.raises(CorruptImageError):
        _open(empty_dir)


def test_group_without_multiscales_is_unsupported(tmp_upath: UPath) -> None:
    group_path = tmp_upath / "plain_group"
    group_path.mkdir(parents=True)
    (group_path / ".zgroup").write_text('{"zarr_format": 2}')
    (group_path / ".zattrs").write_text("{}")
    with pytest.raises(UnsupportedImageFormatError):
        _open(group_path)


def test_corrupt_metadata_json_raises_corrupt_image_error(tmp_upath: UPath) -> None:
    array_path = tmp_upath / "corrupt"
    array_path.mkdir(parents=True)
    (array_path / "zarr.json").write_text("{not valid json")
    with pytest.raises(CorruptImageError):
        _open(array_path)


@pytest.mark.parametrize("writer", [write_ome_zarr_v2_group, write_ome_zarr_v3_group])
def test_unsupported_ome_version_is_rejected(tmp_upath: UPath, writer: Any) -> None:
    group_path = tmp_upath / "future_version"
    data = np.zeros((4, 8, 8), dtype=np.uint8)
    writer(group_path, [("0", data, [1.0, 1.0, 1.0])], _AXES_CZYX[1:])
    # Overwrite the version the writer set with an unsupported one, wherever
    # it lives for that Zarr version (multiscale entry for v2, "ome" object
    # for v3).
    if (group_path / ".zattrs").is_file():
        metadata_path = group_path / ".zattrs"
        metadata = json.loads(metadata_path.read_bytes())
        metadata["multiscales"][0]["version"] = "0.1"
    else:
        metadata_path = group_path / "zarr.json"
        metadata = json.loads(metadata_path.read_bytes())
        metadata["attributes"]["ome"]["version"] = "0.1"
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(UnsupportedImageFormatError, match="0.1"):
        _open(group_path)
