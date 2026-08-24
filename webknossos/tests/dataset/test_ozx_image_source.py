import json
import zipfile
from typing import Any

import numpy as np
import pytest
import tensorstore as ts
from upath import UPath

from tests.data_fixtures import download_wklibs_sample_archive, write_ozx_file
from webknossos.dataset import CorruptImageError, UnsupportedImageFormatError
from webknossos.dataset._image_conversion.image_source import ReadOptions
from webknossos.dataset._image_conversion.ozx_image_source import OzxImageSource


def _open(path: UPath, **options: Any) -> OzxImageSource:
    return OzxImageSource(path, ReadOptions(**options))


# Standard OME axes metadata (name/type), reused across the tests.
_AXES_CZYX = [
    {"name": "c", "type": "channel"},
    {"name": "z", "type": "space"},
    {"name": "y", "type": "space"},
    {"name": "x", "type": "space"},
]


def test_ozx_group_picks_finest_resolution(tmp_upath: UPath) -> None:
    ozx_path = tmp_upath / "multiscale.ozx"
    finest = np.arange(2 * 4 * 8 * 8, dtype=np.uint8).reshape(2, 4, 8, 8)
    coarse = np.zeros((2, 4, 4, 4), dtype=np.uint8)
    write_ozx_file(
        ozx_path,
        [
            ("1", coarse, [1.0, 1.0, 2.0, 2.0]),  # listed first, but coarser
            ("0", finest, [1.0, 1.0, 1.0, 1.0]),
        ],
        _AXES_CZYX,
    )

    source = _open(ozx_path)

    assert (source._z, source._y, source._x) == (4, 8, 8)
    assert source.num_channels == 1
    assert source.get_layer_split_options() == {"channel": [0, 1], "scale": [0, 1]}

    block = source._read_source_box(
        timepoint=0, z=slice(0, 4), y=slice(0, 8), x=slice(0, 8)
    )
    np.testing.assert_array_equal(block[0], finest[0])  # channel 0, the default pin


def test_ozx_group_scale_option_picks_other_level(tmp_upath: UPath) -> None:
    ozx_path = tmp_upath / "multiscale_pick.ozx"
    finest = np.ones((4, 8, 8), dtype=np.uint8)
    coarse = np.full((4, 4, 4), 7, dtype=np.uint8)
    axes = _AXES_CZYX[1:]  # z, y, x only — no channel axis in this group
    write_ozx_file(
        ozx_path,
        [("0", finest, [1.0, 1.0, 1.0]), ("1", coarse, [1.0, 2.0, 2.0])],
        axes,
    )

    finest_source = _open(ozx_path)
    assert (finest_source._z, finest_source._y, finest_source._x) == (4, 8, 8)

    coarse_source = _open(ozx_path, format_options={"scale": 1})
    assert (coarse_source._z, coarse_source._y, coarse_source._x) == (4, 4, 4)
    block = coarse_source._read_source_box(
        timepoint=0, z=slice(0, 4), y=slice(0, 4), x=slice(0, 4)
    )
    np.testing.assert_array_equal(block[0], coarse)


def test_single_resolution_ozx_group_reports_no_scale_choice(tmp_upath: UPath) -> None:
    ozx_path = tmp_upath / "single_scale.ozx"
    data = np.ones((4, 8, 8), dtype=np.uint8)
    write_ozx_file(ozx_path, [("0", data, [1.0, 1.0, 1.0])], _AXES_CZYX[1:])

    source = _open(ozx_path)
    assert source.get_layer_split_options() is None


def test_invalid_scale_option_raises(tmp_upath: UPath) -> None:
    ozx_path = tmp_upath / "single_scale.ozx"
    data = np.ones((4, 8, 8), dtype=np.uint8)
    write_ozx_file(ozx_path, [("0", data, [1.0, 1.0, 1.0])], _AXES_CZYX[1:])

    with pytest.raises(ValueError, match="scale 1 does not exist"):
        _open(ozx_path, format_options={"scale": 1})


def test_remote_path_is_rejected() -> None:
    with pytest.raises(ValueError, match="local file path"):
        _open(UPath("memory://some/path.ozx"))


def test_not_a_zip_file_raises_corrupt_image_error(tmp_upath: UPath) -> None:
    not_a_zip = tmp_upath / "not_a_zip.ozx"
    not_a_zip.write_text("this is not a zip file")
    with pytest.raises(CorruptImageError):
        _open(not_a_zip)


def test_zip_without_root_zarr_json_is_unsupported(tmp_upath: UPath) -> None:
    ozx_path = tmp_upath / "no_root.ozx"
    with zipfile.ZipFile(str(ozx_path), "w") as zip_file:
        zip_file.writestr("some/other/file.txt", "hello")
    with pytest.raises(UnsupportedImageFormatError, match="zarr.json"):
        _open(ozx_path)


def test_zip_with_plain_array_at_root_is_unsupported(tmp_upath: UPath) -> None:
    ozx_path = tmp_upath / "bare_array.ozx"
    with zipfile.ZipFile(str(ozx_path), "w") as zip_file:
        zip_file.writestr(
            "zarr.json",
            json.dumps({"zarr_format": 3, "node_type": "array"}),
        )
    with pytest.raises(UnsupportedImageFormatError, match="multiscale group"):
        _open(ozx_path)


def test_group_without_multiscales_is_unsupported(tmp_upath: UPath) -> None:
    ozx_path = tmp_upath / "plain_group.ozx"
    with zipfile.ZipFile(str(ozx_path), "w") as zip_file:
        zip_file.writestr(
            "zarr.json",
            json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {}}),
        )
    with pytest.raises(UnsupportedImageFormatError, match="multiscales"):
        _open(ozx_path)


def test_corrupt_root_metadata_json_raises_corrupt_image_error(
    tmp_upath: UPath,
) -> None:
    ozx_path = tmp_upath / "corrupt.ozx"
    with zipfile.ZipFile(str(ozx_path), "w") as zip_file:
        zip_file.writestr("zarr.json", "{not valid json")
    with pytest.raises(CorruptImageError):
        _open(ozx_path)


def test_unsupported_ome_version_is_rejected(tmp_upath: UPath) -> None:
    ozx_path = tmp_upath / "future_version.ozx"
    data = np.zeros((4, 8, 8), dtype=np.uint8)
    write_ozx_file(ozx_path, [("0", data, [1.0, 1.0, 1.0])], _AXES_CZYX[1:])

    # Rewrite the archive with the version metadata patched to something
    # unsupported.
    with zipfile.ZipFile(str(ozx_path)) as zip_file:
        metadata = json.loads(zip_file.read("zarr.json"))
        other_entries = {
            info.filename: zip_file.read(info.filename)
            for info in zip_file.infolist()
            if info.filename != "zarr.json"
        }
    metadata["attributes"]["ome"]["version"] = "0.1"
    with zipfile.ZipFile(str(ozx_path), "w") as zip_file:
        zip_file.writestr("zarr.json", json.dumps(metadata))
        for name, content in other_entries.items():
            zip_file.writestr(name, content)

    with pytest.raises(UnsupportedImageFormatError, match="0.1"):
        _open(ozx_path)


def test_axis_names_come_from_ome_axes_metadata(tmp_upath: UPath) -> None:
    # axes in a non-canonical physical order (c, x, y, z) proves axis roles
    # come from the OME metadata, not from position.
    ozx_path = tmp_upath / "labeled.ozx"
    axes = [
        {"name": "c", "type": "channel"},
        {"name": "x", "type": "space"},
        {"name": "y", "type": "space"},
        {"name": "z", "type": "space"},
    ]
    c, x, y, z = 1, 7, 5, 3
    data = np.arange(c * x * y * z, dtype=np.uint8).reshape(c, x, y, z)
    write_ozx_file(ozx_path, [("0", data, [1.0, 1.0, 1.0, 1.0])], axes)

    source = _open(ozx_path)

    assert (source._x, source._y, source._z) == (7, 5, 3)
    block = source._read_source_box(
        timepoint=0, z=slice(0, z), y=slice(0, y), x=slice(0, x)
    )
    assert block.shape == (1, z, y, x)
    np.testing.assert_array_equal(block[0], data[0].transpose(2, 1, 0))


def test_real_world_ozx_sample() -> None:
    # A genuine RFC-9 archive (root zarr.json first, three multiscale levels,
    # two channels), rather than one of this file's own synthetic fixtures.
    ozx_path = download_wklibs_sample_archive("6001240.ozx")

    source = _open(ozx_path)

    assert source.dtype == np.dtype("uint16")
    assert source.num_channels == 1  # two raw channels, one pinned by default
    assert source.get_layer_split_options() == {"channel": [0, 1], "scale": [0, 1, 2]}
    assert (source._z, source._y, source._x) == (236, 275, 271)

    block = source._read_source_box(
        timepoint=0, z=slice(0, 4), y=slice(0, 8), x=slice(0, 8)
    )
    reference = np.asarray(
        ts.open(
            {
                "driver": "zarr3",
                "kvstore": {
                    "driver": "zip",
                    "base": {"driver": "file", "path": str(ozx_path)},
                    "path": "0/",
                },
            },
            open=True,
            context=ts.Context(),
        )
        .result()[0, 0:4, 0:8, 0:8]
        .read()
        .result()
    )
    np.testing.assert_array_equal(block[0], reference)
