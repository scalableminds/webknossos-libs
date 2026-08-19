import json
import zipfile

import numpy as np
import pytest
from upath import UPath

from tests.data_fixtures import download_wklibs_sample_archive
from webknossos import COLOR_CATEGORY, Dataset, Layer
from webknossos.dataset._utils.tensorstore_helpers import read_zarr3_array
from webknossos.geometry import BoundingBox, Mag, NDBoundingBox


def make_layer(dataset_path: UPath) -> tuple[Dataset, Layer, np.ndarray]:
    dataset = Dataset(dataset_path / "test_layer_export", voxel_size=(11, 11, 25))
    layer = dataset.add_layer(
        "color",
        COLOR_CATEGORY,
        dtype="uint8",
        data_format="zarr3",
        bounding_box=BoundingBox((0, 0, 0), (64, 64, 64)),
    )
    np.random.seed(1234)
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
    np.random.seed(1234)
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


def make_odd_sized_layer(tmp_upath: UPath) -> tuple[Dataset, Layer, np.ndarray]:
    """A layer whose size (65) isn't an exact multiple of any of its
    coarser mags' factors, so mag 2 (33**3) and mag 4 (17**3) each have one
    more voxel per axis than floor-dividing 65 would suggest.
    """
    dataset = Dataset(tmp_upath / "test_odd_sized", voxel_size=(10, 10, 10))
    layer = dataset.add_layer(
        "color",
        COLOR_CATEGORY,
        dtype="uint8",
        data_format="zarr3",
        bounding_box=BoundingBox((0, 0, 0), (65, 65, 65)),
    )
    np.random.seed(1234)
    data = (np.random.rand(65, 65, 65) * 255).astype(np.uint8)
    layer.add_mag(1).write(data=data)
    layer.downsample()  # produces mag 2 (33**3) and mag 4 (17**3)
    return dataset, layer, data


def test_as_ozx_odd_sized_layer_covers_full_coarser_mags(tmp_upath: UPath) -> None:
    """Regression test: for a layer whose size isn't an exact multiple of a
    mag's factor, every exported mag must match that mag's full, real data
    (as read directly from the source layer) - not silently drop the last
    row/column/slice, which floor-aligning the exported region used to do.
    """
    _dataset, layer, _data = make_odd_sized_layer(tmp_upath)

    zip_path = tmp_upath / "color_odd.ozx"
    layer.export.as_ozx(output_path=zip_path)

    with zipfile.ZipFile(str(zip_path)) as zip_file:
        zip_file.extractall(str(tmp_upath / "extracted"))

    for mag_name in ("1", "2", "4"):
        expected = layer.get_mag(mag_name).read()
        got = read_zarr3_array(tmp_upath / "extracted" / mag_name)
        got = got[tuple(slice(0, s) for s in expected.shape)]
        assert np.array_equal(got, expected), f"mismatch at mag {mag_name}"


def test_as_tiff_stack_and_as_ome_tiff_odd_sized_layer_cover_full_coarser_mag(
    tmp_upath: UPath,
) -> None:
    """Same regression as test_as_ozx_odd_sized_layer_covers_full_coarser_mags,
    for as_tiff_stack and as_ome_tiff: exporting a coarser mag of a layer
    whose size isn't an exact multiple of that mag's factor must include the
    full, real data (mag 2 is 33**3 here) - not silently drop the last
    row/column/slice.
    """
    pytest.importorskip("tifffile")
    import tifffile

    _dataset, layer, _data = make_odd_sized_layer(tmp_upath)
    expected = layer.get_mag(2).read()  # (1, 33, 33, 33)

    tiff_dir = tmp_upath / "odd_tiff_stack"
    layer.export.as_tiff_stack(output_path=tiff_dir, mag=Mag(2))
    files = sorted(tiff_dir.glob("*.tiff"))
    assert len(files) == expected.shape[3]
    for z, file in enumerate(files):
        image = tifffile.imread(str(file))
        assert np.array_equal(image, expected[0, :, :, z].transpose((1, 0)))

    ome_tiff_path = tmp_upath / "odd.ome.tif"
    layer.export.as_ome_tiff(output_path=ome_tiff_path, mag=Mag(2))
    with tifffile.TiffFile(str(ome_tiff_path)) as tif:
        arr = tif.series[0].asarray()
        assert np.array_equal(arr, expected[0].transpose((2, 1, 0)))


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


def make_nd_layer() -> Layer:
    """Opens the ND (c,t,z,y,x) sample dataset."""
    source_path = download_wklibs_sample_archive("4D") / "4D_series_zarr3"
    dataset = Dataset.open(source_path)
    return dataset.get_layer("color")


def test_as_ozx_nd_layer_ome_metadata(tmp_upath: UPath) -> None:
    layer = make_nd_layer()
    zip_path = tmp_upath / "color_4d.ozx"

    layer.export.as_ozx(output_path=zip_path)

    with zipfile.ZipFile(str(zip_path)) as zip_file:
        root_attrs = json.loads(zip_file.read("zarr.json"))["attributes"]
        axes = root_attrs["ome"]["multiscales"][0]["axes"]
        assert [a["name"] for a in axes] == ["c", "t", "z", "y", "x"]

        mag_shape = json.loads(zip_file.read("1/zarr.json"))["shape"]
        # the OME axes count must match the exported array's ndim
        assert len(axes) == len(mag_shape)


def test_as_tiff_stack_nd_layer(tmp_upath: UPath) -> None:
    pytest.importorskip("tifffile")
    import tifffile

    layer = make_nd_layer()
    data = layer.get_finest_mag().read()  # (c=1, t=7, z=5, y=167, x=439)
    out_dir = tmp_upath / "nd_tiff_stack"

    layer.export.as_tiff_stack(output_path=out_dir)

    files = sorted(out_dir.glob("*.tiff"))
    assert len(files) == 7 * 5

    for t, z in [(0, 0), (3, 2), (6, 4)]:
        image = tifffile.imread(str(out_dir / f"t{t}_z{z}.tiff"))
        expected = data[0, t, z]
        assert np.array_equal(image, expected)


def test_as_ome_tiff_nd_layer_roundtrip(tmp_upath: UPath) -> None:
    pytest.importorskip("tifffile")
    import tifffile

    layer = make_nd_layer()
    data = layer.get_finest_mag().read()  # (c=1, t=7, z=5, y=167, x=439)
    out_path = tmp_upath / "color_4d.ome.tif"

    layer.export.as_ome_tiff(output_path=out_path)

    with tifffile.TiffFile(str(out_path)) as tif:
        arr = tif.series[0].asarray()
        # tifffile drops the size-1 channel axis on read-back
        assert arr.shape == data.shape[1:]
        assert np.array_equal(arr, data[0])


def test_as_ome_tiff_unsupported_axis_raises(tmp_upath: UPath) -> None:
    pytest.importorskip("tifffile")

    dataset = Dataset(tmp_upath / "bad_axis", voxel_size=(10, 10, 10))
    bbox = NDBoundingBox(
        (0, 0, 0, 0), (2, 4, 4, 4), axes=("w", "x", "y", "z"), index=(0, 1, 2, 3)
    )
    layer = dataset.add_layer(
        "color", COLOR_CATEGORY, dtype="uint8", data_format="zarr3", bounding_box=bbox
    )
    layer.add_mag(1).write(
        data=np.zeros((2, 4, 4, 4), dtype="uint8"), absolute_bounding_box=bbox
    )

    with pytest.raises(ValueError, match="w"):
        layer.export.as_ome_tiff(output_path=tmp_upath / "bad.ome.tif")


def test_as_tiff_stack_nd_layer_no_channel_axis(tmp_upath: UPath) -> None:
    pytest.importorskip("tifffile")
    import tifffile

    dataset = Dataset(tmp_upath / "no_channel_axis", voxel_size=(10, 10, 10))
    bbox = NDBoundingBox(
        (0, 0, 0, 0), (2, 3, 4, 5), axes=("t", "z", "y", "x"), index=(0, 1, 2, 3)
    )
    layer = dataset.add_layer(
        "color", COLOR_CATEGORY, dtype="uint8", data_format="zarr3", bounding_box=bbox
    )
    np.random.seed(1234)
    data = (np.random.rand(2, 3, 4, 5) * 255).astype(np.uint8)
    layer.add_mag(1).write(data=data, absolute_bounding_box=bbox)
    out_dir = tmp_upath / "no_channel_tiffs"

    layer.export.as_tiff_stack(output_path=out_dir)

    files = sorted(out_dir.glob("*.tiff"))
    assert len(files) == 2 * 3
    for t in range(2):
        for z in range(3):
            image = tifffile.imread(str(out_dir / f"t{t}_z{z}.tiff"))
            assert np.array_equal(image, data[t, z])
