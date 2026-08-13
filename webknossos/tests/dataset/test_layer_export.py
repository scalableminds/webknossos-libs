import json
import zipfile

import numpy as np
import pytest
from upath import UPath

from webknossos import COLOR_CATEGORY, Dataset, Layer
from webknossos.geometry import BoundingBox


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

        # webknossos Zarr arrays always span from voxel 0, so the exported
        # array's declared shape is the crop's bottomright (32,32,32),
        # rounded up to the small shard shape as_ozx uses (32) - not the
        # original 64**3 layer extent.
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


def test_as_tiff_stack_pixel_values(tmp_upath: UPath) -> None:
    pytest.importorskip("tifffile")
    import tifffile

    _dataset, layer, data = make_layer(tmp_upath)
    crop = BoundingBox((8, 8, 8), (16, 16, 32))
    out_dir = tmp_upath / "tiff_stack"

    layer.export.as_tiff_stack(output_path=out_dir, bounding_box=crop)

    files = sorted(out_dir.glob("*.tiff"))
    assert len(files) == 32

    for z, file in enumerate(files):
        image = tifffile.imread(str(file))
        expected = data[8 : 8 + 16, 8 : 8 + 16, 8 + z].transpose((1, 0))
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


def test_as_ome_tiff_readable(tmp_upath: UPath) -> None:
    pytest.importorskip("tifffile")
    import tifffile

    _dataset, layer, data = make_layer(tmp_upath)
    crop = BoundingBox((8, 8, 8), (16, 16, 32))
    out_path = tmp_upath / "color.ome.tif"

    layer.export.as_ome_tiff(output_path=out_path, bounding_box=crop)

    with tifffile.TiffFile(str(out_path)) as tif:
        arr = tif.series[0].asarray()
        expected = data[8 : 8 + 16, 8 : 8 + 16, 8 : 8 + 32].transpose((2, 1, 0))
        assert arr.shape == expected.shape
        assert np.array_equal(arr, expected)
