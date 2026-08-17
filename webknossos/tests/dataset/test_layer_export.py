import json
import zipfile

import numpy as np
import pytest
from upath import UPath

from webknossos import COLOR_CATEGORY, Dataset, Layer
from webknossos.geometry import BoundingBox, Mag


def make_layer(dataset_path: UPath) -> tuple[Dataset, Layer, np.ndarray]:
    dataset = Dataset(dataset_path / "test_layer_export", voxel_size=(11, 11, 25))
    layer = dataset.add_layer(
        "color",
        COLOR_CATEGORY,
        dtype="uint8",
        data_format="zarr3",
        bounding_box=BoundingBox((0, 0, 0), (64, 64, 64)),
    )
    data = (np.random.rand(64, 64, 64) * 255).astype(np.uint8)
    layer.add_mag(1).write(data=data)
    layer.downsample()
    return dataset, layer, data


def make_layer_with_mags(
    dataset_path: UPath,
) -> tuple[Dataset, Layer, dict[Mag, np.ndarray]]:
    """Like `make_layer`, but with mags "1" and "2-2-2" written with their
    own, independent data, so tests can verify export methods pick the
    right mag rather than always defaulting to the finest one.
    """
    dataset = Dataset(dataset_path / "test_layer_export_mags", voxel_size=(11, 11, 25))
    layer = dataset.add_layer(
        "color",
        COLOR_CATEGORY,
        dtype="uint8",
        data_format="zarr3",
        bounding_box=BoundingBox((0, 0, 0), (64, 64, 64)),
    )
    data_by_mag = {}
    for mag_name, size in [("1", (64, 64, 64)), ("2-2-2", (32, 32, 32))]:
        data = (np.random.rand(*size) * 255).astype(np.uint8)
        layer.add_mag(mag_name).write(data=data)
        data_by_mag[Mag(mag_name)] = data
    return dataset, layer, data_by_mag


def test_as_ozx_cropped(tmp_upath: UPath) -> None:
    _dataset, layer, data = make_layer(tmp_upath)
    crop = BoundingBox((16, 16, 16), (16, 16, 16))
    zip_path = tmp_upath / "color.ozx"

    layer.export.as_ozx(output_path=zip_path, bounding_box=crop)

    with zipfile.ZipFile(str(zip_path)) as zip_file:
        entries = zip_file.infolist()
        assert entries[0].filename == "zarr.json"
        assert all(entry.compress_type == zipfile.ZIP_STORED for entry in entries)

        comment = json.loads(zip_file.comment.decode("utf-8"))
        assert comment["ome"]["version"] == "0.5"

        root_attrs = json.loads(zip_file.read("zarr.json"))["attributes"]
        multiscale_paths = {
            d["path"] for d in root_attrs["ome"]["multiscales"][0]["datasets"]
        }
        # mag=None exports the full pyramid
        assert multiscale_paths == {mag.to_layer_name() for mag in layer.mags}

        # webknossos Zarr arrays always span from voxel 0, and the export is
        # translated to origin, so the exported array's declared shape is
        # the crop's own size (16,16,16), rounded up to the small shard
        # shape as_ozx uses (32) - not the original 64**3 layer extent.
        mag1_shape = json.loads(zip_file.read("1/zarr.json"))["shape"]
        assert list(mag1_shape[-3:]) != [64, 64, 64]
        assert list(mag1_shape[-3:]) == [32, 32, 32]


def test_as_ozx_single_mag(tmp_upath: UPath) -> None:
    _dataset, layer, _data = make_layer(tmp_upath)
    zip_path = tmp_upath / "color_mag2.ozx"

    layer.export.as_ozx(output_path=zip_path, mag=layer.get_mag("2-2-1").mag)

    with zipfile.ZipFile(str(zip_path)) as zip_file:
        names = zip_file.namelist()
        assert "1/zarr.json" not in names
        assert "2-2-1/zarr.json" in names


@pytest.mark.parametrize("mag", [None, Mag("2-2-2")])
def test_as_tiff_stack_pixel_values(tmp_upath: UPath, mag: Mag | None) -> None:
    pytest.importorskip("tifffile")
    import tifffile

    _dataset, layer, data_by_mag = make_layer_with_mags(tmp_upath)
    data = data_by_mag[mag or Mag(1)]

    # divisible by 2, so the crop is already aligned for both mag 1 and 2-2-2
    crop = BoundingBox((16, 16, 16), (32, 32, 32))
    out_dir = tmp_upath / "tiff_stack"

    layer.export.as_tiff_stack(output_path=out_dir, bounding_box=crop, mag=mag)

    crop_in_mag = crop.in_mag(mag or Mag(1))
    files = sorted(out_dir.glob("*.tiff"))
    assert len(files) == crop_in_mag.size.z

    for z, file in enumerate(files):
        image = tifffile.imread(str(file))
        expected = data[
            crop_in_mag.topleft.x : crop_in_mag.topleft.x + crop_in_mag.size.x,
            crop_in_mag.topleft.y : crop_in_mag.topleft.y + crop_in_mag.size.y,
            crop_in_mag.topleft.z + z,
        ].transpose((1, 0))
        assert np.array_equal(image, expected)


def test_as_tiff_stack_filename_prefix(tmp_upath: UPath) -> None:
    pytest.importorskip("tifffile")

    _dataset, layer, _data = make_layer(tmp_upath)
    crop = BoundingBox((8, 8, 8), (4, 4, 3))
    out_dir = tmp_upath / "tiff_stack_prefixed"

    layer.export.as_tiff_stack(
        output_path=out_dir, bounding_box=crop, filename_prefix="section"
    )

    files = sorted(f.name for f in out_dir.glob("*.tiff"))
    # 3 slices -> single-digit indices are enough.
    assert files == ["section_0.tiff", "section_1.tiff", "section_2.tiff"]


@pytest.mark.parametrize("mag", [None, Mag("2-2-2")])
def test_as_ome_tiff_readable(tmp_upath: UPath, mag: Mag | None) -> None:
    pytest.importorskip("tifffile")
    import tifffile

    _dataset, layer, data_by_mag = make_layer_with_mags(tmp_upath)
    data = data_by_mag[mag or Mag(1)]

    crop = BoundingBox((16, 16, 16), (32, 32, 32))
    out_path = tmp_upath / "color.ome.tif"

    layer.export.as_ome_tiff(output_path=out_path, bounding_box=crop, mag=mag)

    crop_in_mag = crop.in_mag(mag or Mag(1))
    with tifffile.TiffFile(str(out_path)) as tif:
        arr = tif.series[0].asarray()
        expected = data[
            crop_in_mag.topleft.x : crop_in_mag.topleft.x + crop_in_mag.size.x,
            crop_in_mag.topleft.y : crop_in_mag.topleft.y + crop_in_mag.size.y,
            crop_in_mag.topleft.z : crop_in_mag.topleft.z + crop_in_mag.size.z,
        ].transpose((2, 1, 0))
        assert arr.shape == expected.shape
        assert np.array_equal(arr, expected)
