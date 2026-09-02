import pickle

import numpy as np
import pytest
from upath import UPath

from tests.constants import (
    REMOTE_TESTOUTPUT_DIR,
    TESTOUTPUT_DIR,
)
from tests.dataset._dataset_helpers import (
    DATA_FORMATS_AND_OUTPUT_PATHS,
    assure_exported_properties,
    copy_simple_dataset,
    get_multichanneled_data,
    prepare_dataset_path,
)
from webknossos.dataset import (
    Dataset,
)
from webknossos.dataset_properties import (
    COLOR_CATEGORY,
    SEGMENTATION_CATEGORY,
    DataFormat,
)
from webknossos.geometry import (
    BoundingBox,
    Mag,
    NDBoundingBox,
    Vec3Int,
    VecIntLike,
)
from webknossos.geometry.constants import (
    C_AXIS,
    T_AXIS,
    X_AXIS,
    Y_AXIS,
    Z_AXIS,
)

rng = np.random.default_rng(1234)

pytestmark = pytest.mark.usefixtures("moto_server")


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_view_read(data_format: DataFormat, output_path: UPath) -> None:
    ds_path = copy_simple_dataset(data_format, output_path)

    with pytest.warns(UserWarning, match=".*not aligned with the shard shape.*"):
        wk_view = (
            Dataset.open(ds_path)
            .get_layer("color")
            .get_mag("1")
            .get_view(absolute_offset=(0, 0, 0), size=(16, 16, 16))
        )

    # 'read()' checks if it was already opened. If not, it opens it automatically
    data = wk_view.read(absolute_offset=(0, 0, 0), size=(10, 10, 10))
    assert data.shape == (3, 10, 10, 10)  # three channel
    assert wk_view.info.data_format == data_format


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_view_write(data_format: DataFormat, output_path: UPath) -> None:
    ds_path = copy_simple_dataset(data_format, output_path)
    with pytest.warns(UserWarning, match=".*not aligned with the shard shape.*"):
        wk_view = (
            Dataset.open(ds_path)
            .get_layer("color")
            .get_mag("1")
            .get_view(absolute_offset=(0, 0, 0), size=(16, 16, 16))
        )

    assert wk_view.info.data_format == data_format

    write_data = rng.integers(0, 256, (3, 10, 10, 10), dtype=np.uint8)

    wk_view.write(write_data, allow_unaligned=True)

    data = wk_view.read(absolute_offset=(0, 0, 0), size=(10, 10, 10))
    np.testing.assert_array_equal(data, write_data)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_read_cxyz(data_format: DataFormat, output_path: UPath) -> None:
    ds_path = copy_simple_dataset(data_format, output_path)

    with pytest.warns(UserWarning, match=".*not aligned with the shard shape.*"):
        wk_view = (
            Dataset.open(ds_path)
            .get_layer("color")
            .get_mag("1")
            .get_view(absolute_offset=(0, 0, 0), size=(16, 16, 16))
        )

    data_cxyz = wk_view.read_cxyz(absolute_offset=(0, 0, 0), size=(10, 10, 10))
    assert data_cxyz.shape == (3, 10, 10, 10)

    # read_cxyz must return the same data as read() for standard axis ordering
    data = wk_view.read(absolute_offset=(0, 0, 0), size=(10, 10, 10))
    np.testing.assert_array_equal(data_cxyz, data)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_write_cxyz(data_format: DataFormat, output_path: UPath) -> None:
    ds_path = copy_simple_dataset(data_format, output_path)
    with pytest.warns(UserWarning, match=".*not aligned with the shard shape.*"):
        wk_view = (
            Dataset.open(ds_path)
            .get_layer("color")
            .get_mag("1")
            .get_view(absolute_offset=(0, 0, 0), size=(16, 16, 16))
        )

    write_data = rng.integers(0, 256, (3, 10, 10, 10), dtype=np.uint8)

    wk_view.write_cxyz(write_data, allow_unaligned=True)

    data = wk_view.read_cxyz(absolute_offset=(0, 0, 0), size=(10, 10, 10))
    np.testing.assert_array_equal(data, write_data)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_read_cxyz_adds_channel_axis(
    data_format: DataFormat, output_path: UPath
) -> None:
    if data_format == DataFormat.WKW:
        pytest.skip(
            "WKW requires (c, x, y, z) axes and cannot store channel-free layers"
        )
    ds_path = prepare_dataset_path(data_format, output_path)
    layer = Dataset(ds_path, voxel_size=(1, 1, 1)).add_layer(
        "segmentation",
        SEGMENTATION_CATEGORY,
        bounding_box=NDBoundingBox((0, 0, 0), (10, 10, 10), axes="xyz"),
        data_format=data_format,
        num_channels=1,
    )
    mag = layer.add_mag("1")

    write_data = np.zeros((10, 10, 10), dtype=np.uint64)
    mag.write(write_data, absolute_offset=(0, 0, 0))

    data = mag.read(absolute_offset=(0, 0, 0), size=(10, 10, 10))
    assert data.shape == (10, 10, 10)

    data = mag.read_cxyz(absolute_offset=(0, 0, 0), size=(10, 10, 10))
    assert data.shape == (1, 10, 10, 10)

    # Passing absolute_bounding_box=BoundingBox must not raise a rank mismatch
    data = mag.read_cxyz(absolute_bounding_box=BoundingBox((0, 0, 0), (10, 10, 10)))
    assert data.shape == (1, 10, 10, 10)

    # Same for write_cxyz
    write_data_cxyz = np.ones((1, 10, 10, 10), dtype=np.uint64)
    mag.write_cxyz(
        write_data_cxyz,
        allow_unaligned=True,
        absolute_bounding_box=BoundingBox((0, 0, 0), (10, 10, 10)),
    )
    readback = mag.read_cxyz(absolute_bounding_box=BoundingBox((0, 0, 0), (10, 10, 10)))
    np.testing.assert_array_equal(readback, write_data_cxyz)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_read_write_cxyz_bounding_box_with_extra_axes(
    data_format: DataFormat, output_path: UPath
) -> None:
    """read_cxyz/write_cxyz with BoundingBox must work for layers that have extra axes (e.g. t) with size 1."""
    if data_format == DataFormat.WKW:
        pytest.skip("WKW requires (c, x, y, z) axes and cannot store extra axes")
    ds_path = prepare_dataset_path(data_format, output_path)
    layer = Dataset(ds_path, voxel_size=(1, 1, 1)).add_layer(
        "segmentation",
        SEGMENTATION_CATEGORY,
        bounding_box=NDBoundingBox(
            (0, 0, 0, 0),
            (10, 10, 10, 1),
            axes=(X_AXIS, Y_AXIS, Z_AXIS, T_AXIS),
            index=(0, 1, 2, 3),
        ),
        data_format=data_format,
        num_channels=1,
    )
    mag = layer.add_mag("1")
    layer_bbox = NDBoundingBox(
        (0, 0, 0, 0),
        (10, 10, 10, 1),
        axes=(X_AXIS, Y_AXIS, Z_AXIS, T_AXIS),
        index=(0, 1, 2, 3),
    )

    # read_cxyz with BoundingBox must not raise a rank mismatch with the 4D (x,y,z,t) array
    data_bbox = mag.read_cxyz(
        absolute_bounding_box=BoundingBox((0, 0, 0), (10, 10, 10))
    )
    assert data_bbox.shape == (1, 10, 10, 10)

    # Reading via BoundingBox and via NDBoundingBox must produce identical results
    data_ndbbox = mag.read_cxyz(absolute_bounding_box=layer_bbox)
    np.testing.assert_array_equal(data_bbox, data_ndbbox)

    # write_cxyz with BoundingBox must not raise; verify with uniform data to avoid
    # pre-existing axis-ordering quirks in the 5D write path
    write_data_cxyz = np.ones((1, 10, 10, 10), dtype=np.uint64)
    mag.write_cxyz(
        write_data_cxyz,
        allow_unaligned=True,
        absolute_bounding_box=BoundingBox((0, 0, 0), (10, 10, 10)),
    )
    readback = mag.read_cxyz(absolute_bounding_box=layer_bbox)
    np.testing.assert_array_equal(readback, write_data_cxyz)


@pytest.mark.parametrize(
    "layer_bbox,write_bbox,write_data,expected_shape",
    [
        (
            NDBoundingBox(
                topleft=(0, 0), size=(10, 20), axes=(X_AXIS, Y_AXIS), index=(0, 1)
            ),
            NDBoundingBox(
                topleft=(0, 0), size=(10, 20), axes=(X_AXIS, Y_AXIS), index=(0, 1)
            ),
            np.arange(200, dtype=np.uint8).reshape(1, 10, 20, 1),
            (1, 10, 20, 1),
        ),
        (
            NDBoundingBox(
                topleft=(0, 0, 0, 0, 0),
                size=(1, 4, 4, 4, 2),
                axes=(C_AXIS, X_AXIS, Y_AXIS, Z_AXIS, T_AXIS),
                index=(0, 1, 2, 3, 4),
            ),
            NDBoundingBox(
                topleft=(0, 0, 0, 0, 0),
                size=(1, 4, 4, 4, 1),
                axes=(C_AXIS, X_AXIS, Y_AXIS, Z_AXIS, T_AXIS),
                index=(0, 1, 2, 3, 4),
            ),
            np.zeros((1, 4, 4, 4), dtype=np.uint8),
            (1, 4, 4, 4),
        ),
        (
            NDBoundingBox(
                topleft=(0, 0, 0),
                size=(10, 20, 5),
                axes=(X_AXIS, Y_AXIS, Z_AXIS),
                index=(0, 1, 2),
            ),
            NDBoundingBox(
                topleft=(0, 0, 0),
                size=(10, 20, 5),
                axes=(X_AXIS, Y_AXIS, Z_AXIS),
                index=(0, 1, 2),
            ),
            (np.arange(1000, dtype=np.uint8)).reshape(1, 10, 20, 5),
            (1, 10, 20, 5),
        ),
    ],
)
def test_read_write_cxyz_axes(
    tmp_path: UPath,
    layer_bbox: NDBoundingBox,
    write_bbox: NDBoundingBox,
    write_data: np.ndarray,
    expected_shape: tuple[int, ...],
) -> None:
    """read_cxyz/write_cxyz must work when the bounding box has no z axis."""
    mag = (
        Dataset(tmp_path / "ds", voxel_size=(1, 1, 1))
        .add_layer("color", COLOR_CATEGORY, bounding_box=layer_bbox)
        .add_mag("1")
    )

    mag.write_cxyz(write_data, absolute_bounding_box=write_bbox)

    data = mag.read_cxyz(absolute_bounding_box=write_bbox)
    assert data.shape == expected_shape, f"unexpected shape {data.shape}"
    np.testing.assert_array_equal(data, write_data)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_write_cxyz_mag_view(data_format: DataFormat, output_path: UPath) -> None:
    ds_path = prepare_dataset_path(data_format, output_path)
    layer = Dataset(ds_path, voxel_size=(1, 1, 1)).add_layer(
        "color", COLOR_CATEGORY, num_channels=3
    )
    mag = layer.add_mag("1")

    write_data = rng.integers(0, 256, (3, 10, 10, 10), dtype=np.uint8)

    # without allow_resize should fail
    with pytest.raises(
        ValueError, match=".*does not fit in the layer's bounding box.*"
    ):
        mag.write_cxyz(write_data, absolute_offset=(0, 0, 0))

    # with allow_resize should succeed and update the bounding box
    mag.write_cxyz(write_data, absolute_offset=(0, 0, 0), allow_resize=True)

    assert layer.bounding_box == BoundingBox((0, 0, 0), (10, 10, 10))
    data = mag.read_cxyz(absolute_offset=(0, 0, 0), size=(10, 10, 10))
    np.testing.assert_array_equal(data, write_data)


@pytest.mark.parametrize("output_path", [TESTOUTPUT_DIR, REMOTE_TESTOUTPUT_DIR])
@pytest.mark.parametrize("data_format", [DataFormat.Zarr, DataFormat.Zarr3])
def test_direct_zarr_access(output_path: UPath, data_format: DataFormat) -> None:
    ds_path = copy_simple_dataset(data_format, output_path)
    mag = Dataset.open(ds_path).get_layer("color").get_mag("1")

    # write: zarr, read: wk
    write_data = rng.integers(0, 256, (3, 10, 10, 10), dtype=np.uint8)
    mag.get_zarr_array()[:, 0:10, 0:10, 0:10].write(write_data).result()
    data = mag.read(absolute_offset=(0, 0, 0), size=(10, 10, 10))
    np.testing.assert_array_equal(data, write_data)

    # write: wk, read: zarr
    write_data = rng.integers(0, 256, (3, 10, 10, 10), dtype=np.uint8)
    mag.write(write_data, absolute_offset=(0, 0, 0), allow_unaligned=True)
    data = mag.get_zarr_array()[:, 0:10, 0:10, 0:10].read().result()
    np.testing.assert_array_equal(data, write_data)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_view_write_out_of_bounds(data_format: DataFormat, output_path: UPath) -> None:
    ds_path = copy_simple_dataset(
        data_format, output_path, "view_dataset_out_of_bounds"
    )

    with pytest.warns(UserWarning, match=".*not aligned with the shard shape.*"):
        view = (
            Dataset.open(ds_path)
            .get_layer("color")
            .get_mag("1")
            .get_view(absolute_offset=(0, 0, 0), size=(16, 16, 16))
        )

    with pytest.raises(AssertionError):
        view.write(
            np.zeros((200, 200, 5), dtype=np.uint8)
        )  # this is bigger than the bounding_box


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_mag_view_write_out_of_bounds(
    data_format: DataFormat, output_path: UPath
) -> None:
    ds_path = copy_simple_dataset(data_format, output_path, "dataset_out_of_bounds")

    ds = Dataset.open(ds_path)
    mag_view = ds.get_layer("color").get_mag("1")

    assert mag_view.info.data_format == data_format

    assert tuple(ds.get_layer("color").bounding_box.size) == (24, 24, 24)
    mag_view.write(
        np.zeros((3, 1, 1, 48), dtype=np.uint8), allow_resize=True, allow_unaligned=True
    )  # this is bigger than the bounding_box
    assert tuple(ds.get_layer("color").bounding_box.size) == (24, 24, 48)

    assure_exported_properties(ds)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_mag_view_write_out_of_bounds_mag2(
    data_format: DataFormat, output_path: UPath
) -> None:
    ds_path = copy_simple_dataset(data_format, output_path, "dataset_out_of_bounds")

    ds = Dataset.open(ds_path)
    color_layer = ds.get_layer("color")
    mag_view = color_layer.get_or_add_mag("2-2-1", compress=False)

    assert color_layer.bounding_box.topleft == Vec3Int(0, 0, 0)
    assert color_layer.bounding_box.size == Vec3Int(24, 24, 24)
    mag_view.write(
        np.zeros((3, 50, 1, 48), dtype=np.uint8),
        absolute_offset=(20, 20, 10),
        allow_resize=True,
        allow_unaligned=True,
    )  # this is bigger than the bounding_box
    assert color_layer.bounding_box.topleft == Vec3Int(0, 0, 0)
    assert color_layer.bounding_box.size == Vec3Int(120, 24, 58)

    assure_exported_properties(ds)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_view_write_allow_resize(data_format: DataFormat, output_path: UPath) -> None:
    ds_path = prepare_dataset_path(data_format, output_path)
    layer = Dataset(ds_path, voxel_size=(1, 1, 1)).add_layer("color", COLOR_CATEGORY)
    mag = layer.add_mag("1")

    write_data = rng.integers(0, 256, (10, 10, 10), dtype=np.uint8)

    # this should fail
    with pytest.raises(
        ValueError, match=".*does not fit in the layer's bounding box.*"
    ):
        mag.write(absolute_offset=(0, 0, 0), data=write_data)

    # this should go through
    mag.write(absolute_offset=(0, 0, 0), data=write_data, allow_resize=True)

    assert layer.bounding_box == BoundingBox((0, 0, 0), (10, 10, 10))
    data = mag.read(absolute_offset=(0, 0, 0), size=(10, 10, 10)).squeeze(0)
    np.testing.assert_array_equal(data, write_data)

    # override with same bbox
    mag.write(
        absolute_offset=(0, 0, 0),
        data=rng.integers(0, 256, (10, 10, 10), dtype=np.uint8),
    )

    # resize to larger bbox
    mag.write(
        absolute_offset=(10, 10, 10),
        data=rng.integers(0, 256, (5, 5, 5), dtype=np.uint8),
        allow_resize=True,
        allow_unaligned=True,
    )
    assert layer.bounding_box == BoundingBox((0, 0, 0), (15, 15, 15))


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_view_write_allow_unaligned(
    data_format: DataFormat, output_path: UPath
) -> None:
    ds_path = prepare_dataset_path(data_format, output_path)
    layer = Dataset(ds_path, voxel_size=(1, 1, 1)).add_layer(
        "color",
        COLOR_CATEGORY,
        bounding_box=BoundingBox((0, 0, 0), (32, 32, 32)),
        data_format=data_format,
    )
    mag = layer.add_mag(
        "1",
        chunk_shape=(8, 8, 8),
        shard_shape=(8, 8, 8) if data_format == DataFormat.Zarr else (16, 16, 16),
    )

    write_data = rng.integers(0, 256, (4, 4, 4), dtype=np.uint8)

    # this should fail
    with pytest.raises(ValueError, match=".*is not aligned with the shard shape.*"):
        mag.write(absolute_offset=(0, 0, 0), data=write_data)

    # this should go through
    mag.write(absolute_offset=(0, 0, 0), data=write_data, allow_unaligned=True)

    data = mag.read(absolute_offset=(0, 0, 0), size=(4, 4, 4)).squeeze(0)
    np.testing.assert_array_equal(data, write_data)

    # override a whole shard
    mag.write(
        absolute_offset=(16, 16, 16),
        data=rng.integers(0, 256, (16, 16, 16), dtype=np.uint8),
    )

    # override multiple shards
    mag.write(
        absolute_offset=(16, 16, 0),
        data=rng.integers(0, 256, (16, 16, 32), dtype=np.uint8),
    )

    # override the whole bbox
    mag.write(
        absolute_offset=(0, 0, 0),
        data=rng.integers(0, 256, (32, 32, 32), dtype=np.uint8),
    )


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_views_are_equal(data_format: DataFormat, output_path: UPath) -> None:
    data: np.ndarray = rng.integers(0, 256, (10, 10, 10), dtype=np.uint8)

    path_a = prepare_dataset_path(data_format, output_path / "a")
    path_b = prepare_dataset_path(data_format, output_path / "b")
    mag_a = (
        Dataset(path_a, voxel_size=(1, 1, 1))
        .get_or_add_layer(
            "color",
            COLOR_CATEGORY,
            data_format=data_format,
            bounding_box=BoundingBox((0, 0, 0), data.shape),
        )
        .get_or_add_mag("1")
    )
    mag_b = (
        Dataset(path_b, voxel_size=(1, 1, 1))
        .get_or_add_layer(
            "color",
            COLOR_CATEGORY,
            data_format=data_format,
            bounding_box=BoundingBox((0, 0, 0), data.shape),
        )
        .get_or_add_mag("1")
    )

    mag_a.write(data)
    mag_b.write(data)
    assert mag_a.content_is_equal(mag_b, chunk_shape=Vec3Int.full(64))

    data = data + 10
    mag_b.write(data)
    assert not mag_a.content_is_equal(mag_b)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_update_new_bounding_box_offset(
    data_format: DataFormat, output_path: UPath
) -> None:
    ds_path = prepare_dataset_path(data_format, output_path)
    ds = Dataset(ds_path, voxel_size=(1, 1, 1))
    color_layer = ds.add_layer("color", COLOR_CATEGORY, data_format=data_format)
    mag = color_layer.add_mag("1", compress=False)

    assert color_layer.bounding_box.topleft == Vec3Int(0, 0, 0)

    write_data = rng.integers(0, 256, (10, 10, 10), dtype=np.uint8)
    mag.write(
        write_data,
        absolute_offset=(10, 10, 10),
        allow_resize=True,
        allow_unaligned=True,
    )  # the write method of MagDataset does always use the relative offset to (0, 0, 0)
    assert color_layer.bounding_box.topleft == Vec3Int(10, 10, 10)
    assert color_layer.bounding_box.size == Vec3Int(10, 10, 10)

    mag.write(
        write_data, absolute_offset=(5, 5, 20), allow_resize=True, allow_unaligned=True
    )  # the write method of MagDataset does always use the relative offset to (0, 0, 0)
    assert color_layer.bounding_box.topleft == Vec3Int(5, 5, 10)
    assert color_layer.bounding_box.size == Vec3Int(15, 15, 20)

    assure_exported_properties(ds)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_write_multi_channel_uint8(data_format: DataFormat, output_path: UPath) -> None:
    ds_path = prepare_dataset_path(data_format, output_path, "multichannel")
    ds = Dataset(ds_path, voxel_size=(1, 1, 1))
    mag = ds.add_layer(
        "color", COLOR_CATEGORY, num_channels=3, data_format=data_format
    ).add_mag(
        "1", shard_shape=(512, 512, 32) if data_format == DataFormat.Zarr3 else None
    )

    data = get_multichanneled_data(np.uint8)

    mag.write(data, allow_resize=True)

    np.testing.assert_array_equal(data, mag.read())

    assure_exported_properties(ds)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_wkw_write_multi_channel_uint16(
    data_format: DataFormat, output_path: UPath
) -> None:
    ds_path = prepare_dataset_path(data_format, output_path, "multichannel")
    ds = Dataset(ds_path, voxel_size=(1, 1, 1))
    mag = ds.add_layer(
        "color",
        COLOR_CATEGORY,
        num_channels=3,
        dtype="uint16",
        data_format=data_format,
    ).add_mag(
        "1", shard_shape=(512, 512, 32) if data_format == DataFormat.Zarr3 else None
    )

    data = get_multichanneled_data(np.uint16)

    mag.write(data, allow_resize=True)
    written_data = mag.read()

    np.testing.assert_array_equal(data, written_data)

    assure_exported_properties(ds)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_empty_read(data_format: DataFormat, output_path: UPath) -> None:
    ds_path = prepare_dataset_path(data_format, output_path, "empty")
    mag = (
        Dataset(ds_path, voxel_size=(1, 1, 1))
        .add_layer("color", category=COLOR_CATEGORY, data_format=data_format)
        .add_mag("1")
    )
    with pytest.raises(AssertionError):
        # size
        mag.read(absolute_offset=(0, 0, 0), size=(0, 0, 0))


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
@pytest.mark.parametrize("absolute_offset", [None, Vec3Int(12, 12, 12)])
def test_write_layer(
    data_format: DataFormat, output_path: UPath, absolute_offset: Vec3Int | None
) -> None:
    ds_path = prepare_dataset_path(data_format, output_path, "empty")
    ds = Dataset(ds_path, voxel_size=(1, 1, 1))

    data: np.ndarray = rng.integers(0, 256, (128, 128, 128), dtype=np.uint8)
    layer = ds.write_layer(
        "color",
        category=COLOR_CATEGORY,
        data=data,
        data_format=data_format,
        absolute_offset=absolute_offset,
    )

    np.testing.assert_array_equal(layer.get_mag(1).read().squeeze(), data)
    if absolute_offset is not None:
        assert layer.bounding_box.topleft_xyz == absolute_offset
    assert layer.bounding_box.size_xyz == Vec3Int(data.shape)
    assert Mag(2) in layer.mags  # did downsample
    assert Mag(4) in layer.mags  # did downsample


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
@pytest.mark.parametrize("absolute_offset", [None, Vec3Int(12, 12, 12)])
def test_write_layer_mag2(
    data_format: DataFormat, output_path: UPath, absolute_offset: Vec3Int | None
) -> None:
    ds_path = prepare_dataset_path(data_format, output_path, "empty")
    ds = Dataset(ds_path, voxel_size=(12, 12, 24))

    data: np.ndarray = rng.integers(0, 256, (128, 128, 128), dtype=np.uint8)
    layer = ds.write_layer(
        "color",
        category=COLOR_CATEGORY,
        data=data,
        data_format=data_format,
        absolute_offset=absolute_offset,
        mag=(2, 2, 1),
    )

    np.testing.assert_array_equal(layer.get_mag((2, 2, 1)).read().squeeze(), data)
    if absolute_offset is not None:
        assert layer.bounding_box.topleft_xyz == absolute_offset  # in mag1
    assert layer.bounding_box.size_xyz == Vec3Int(data.shape) * Vec3Int(
        2, 2, 1
    )  # in mag1
    assert Mag((4, 4, 2)) in layer.mags  # did downsample


@pytest.mark.parametrize(
    "data_format,output_path",
    [(DataFormat.Zarr3, TESTOUTPUT_DIR), (DataFormat.Zarr3, REMOTE_TESTOUTPUT_DIR)],
)
@pytest.mark.parametrize("absolute_offset", [None, (0, 3, 12, 12, 12)])
def test_write_layer_5d(
    data_format: DataFormat,
    output_path: UPath,
    absolute_offset: VecIntLike | None,
) -> None:
    ds_path = prepare_dataset_path(data_format, output_path, "empty")
    ds = Dataset(ds_path, voxel_size=(1, 1, 1))

    data: np.ndarray = rng.integers(0, 256, (3, 2, 128, 128, 128), dtype=np.uint8)
    layer = ds.write_layer(
        "color",
        category=COLOR_CATEGORY,
        data=data,
        data_format=data_format,
        axes=(C_AXIS, T_AXIS, X_AXIS, Y_AXIS, Z_AXIS),
        shard_shape=(128, 128, 128),
        absolute_offset=absolute_offset,
    )
    np.testing.assert_array_equal(layer.get_mag(1).read().squeeze(), data)
    if absolute_offset is not None:
        assert layer.bounding_box.topleft.to_tuple() == absolute_offset
    assert layer.bounding_box.size.to_tuple() == data.shape


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_read_padded_data(data_format: DataFormat, output_path: UPath) -> None:
    ds_path = prepare_dataset_path(data_format, output_path, "empty")
    mag = (
        Dataset(ds_path, voxel_size=(1, 1, 1))
        .add_layer(
            "color", category=COLOR_CATEGORY, num_channels=3, data_format=data_format
        )
        .add_mag("1")
    )
    # there is no data yet, however, this should not fail but pad the data with zeros
    data = mag.read(absolute_offset=(0, 0, 0), size=(10, 10, 10))

    assert data.shape == (3, 10, 10, 10)
    np.testing.assert_array_equal(data, np.zeros((3, 10, 10, 10)))


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_num_channel_mismatch_assertion(
    data_format: DataFormat, output_path: UPath
) -> None:
    ds_path = prepare_dataset_path(data_format, output_path)
    ds = Dataset(ds_path, voxel_size=(1, 1, 1))
    mag = ds.add_layer(
        "color", category=COLOR_CATEGORY, num_channels=1, data_format=data_format
    ).add_mag("1")  # num_channel=1 is also the default

    write_data = rng.integers(0, 256, (3, 10, 10, 10), dtype=np.uint8)  # 3 channels

    with pytest.raises(AssertionError):
        mag.write(
            write_data, allow_resize=True
        )  # there is a mismatch between the number of channels

    assure_exported_properties(ds)


def test_get_view() -> None:
    ds_path = prepare_dataset_path(DataFormat.WKW, TESTOUTPUT_DIR, "get_view")
    ds = Dataset(ds_path, voxel_size=(1, 1, 1))
    mag = ds.add_layer("color", COLOR_CATEGORY).add_mag("1")

    # The dataset is new -> no data has been written.
    # Therefore, the size of the bounding box in the properties.json is (0, 0, 0)

    # Creating this view works because the size is set to (0, 0, 0)
    # However, in practice a view with size (0, 0, 0) would not make sense
    # Sizes that contain "0" are not allowed usually, except for an empty layer
    assert mag.get_view().bounding_box.is_empty()

    with pytest.raises(AssertionError):
        # This view exceeds the bounding box
        mag.get_view(relative_offset=(0, 0, 0), size=(16, 16, 16))

    # read-only-views may exceed the bounding box
    read_only_view = mag.get_view(
        relative_offset=(0, 0, 0), size=(16, 16, 16), read_only=True
    )
    assert read_only_view.bounding_box == BoundingBox((0, 0, 0), (16, 16, 16))

    with pytest.raises(AssertionError):
        # Trying to get a writable sub-view of a read-only-view is not allowed
        read_only_view.get_view(read_only=False)

    write_data = rng.integers(0, 256, (100, 200, 300), dtype=np.uint8)
    # This operation updates the bounding box of the dataset according to the written data
    mag.write(write_data, absolute_offset=(10, 20, 30), allow_resize=True)

    with pytest.raises(AssertionError):
        # The offset and size default to (0, 0, 0).
        # Sizes that contain "0" are not allowed
        mag.get_view(absolute_offset=(0, 0, 0), size=(10, 10, 0))

    assert mag.bounding_box.bottomright == Vec3Int(110, 220, 330)

    with pytest.warns(UserWarning, match=".*not aligned with the shard shape.*"):
        # Therefore, creating a view with a size of (16, 16, 16) is now allowed
        wk_view = mag.get_view(relative_offset=(0, 0, 0), size=(16, 16, 16))
    assert wk_view.bounding_box == BoundingBox((10, 20, 30), (16, 16, 16))

    with pytest.raises(AssertionError):
        # Creating this view does not work because the offset (0, 0, 0) would be outside
        # of the bounding box from the properties.json.
        mag.get_view(size=(26, 36, 46), absolute_offset=(0, 0, 0))

    # But setting "read_only=True" still works
    mag.get_view(size=(26, 36, 46), absolute_offset=(0, 0, 0), read_only=True)

    with pytest.warns(UserWarning, match=".*not aligned with the shard shape.*"):
        # Creating this subview works because the subview is completely inside the 'wk_view'.
        # Note that the offset in "get_view" is always relative to the "global_offset"-attribute of the called view.
        sub_view = wk_view.get_view(relative_offset=(8, 8, 8), size=(8, 8, 8))
    assert sub_view.bounding_box == BoundingBox((18, 28, 38), (8, 8, 8))

    with pytest.raises(AssertionError):
        # Creating this subview does not work because it is not completely inside the 'wk_view'
        wk_view.get_view(relative_offset=(8, 8, 8), size=(10, 10, 10))

    # Again: read-only is allowed
    wk_view.get_view(relative_offset=(8, 8, 8), size=(10, 10, 10), read_only=True)

    with pytest.raises(AssertionError):
        # negative offsets are not allowed
        mag.get_view(absolute_offset=(-1, -2, -3))

    assure_exported_properties(ds)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_read_only_view(data_format: DataFormat, output_path: UPath) -> None:
    ds_path = prepare_dataset_path(data_format, output_path, "read_only_view")
    ds = Dataset(ds_path, voxel_size=(1, 1, 1))
    mag = ds.get_or_add_layer(
        "color", COLOR_CATEGORY, data_format=data_format
    ).get_or_add_mag("1")
    mag.write(
        data=rng.integers(0, 256, (1, 10, 10, 10), dtype=np.uint8),
        absolute_offset=(10, 20, 30),
        allow_resize=True,
        allow_unaligned=True,
    )
    v_write = mag.get_view()
    v_read = mag.get_view(read_only=True)

    new_data = rng.integers(0, 256, (1, 5, 6, 7), dtype=np.uint8)
    with pytest.raises(RuntimeError):
        v_read.write(data=new_data)

    v_write.write(data=new_data, allow_unaligned=True)

    assure_exported_properties(ds)


def test_read_bbox() -> None:
    ds_path = prepare_dataset_path(DataFormat.WKW, TESTOUTPUT_DIR)
    ds = Dataset(ds_path, voxel_size=(2, 2, 1))
    layer = ds.add_layer("color", COLOR_CATEGORY)
    mag = layer.add_mag(1)
    mag.write(
        absolute_offset=(10, 20, 30),
        data=rng.integers(0, 256, (50, 60, 70), dtype=np.uint8),
        allow_resize=True,
    )

    np.testing.assert_array_equal(
        mag.read(absolute_offset=(20, 30, 40), size=(40, 50, 60)),
        mag.read(
            absolute_bounding_box=BoundingBox(topleft=(20, 30, 40), size=(40, 50, 60))
        ),
    )


def test_pickle_view() -> None:
    ds_path = prepare_dataset_path(DataFormat.WKW, TESTOUTPUT_DIR, "pickle")
    ds = Dataset(ds_path, voxel_size=(1, 1, 1))
    mag1 = ds.add_layer("color", COLOR_CATEGORY).add_mag(1)

    data_to_write = rng.integers(0, 256, (1, 10, 10, 10), dtype=np.uint8)
    mag1.write(data_to_write, allow_resize=True)
    assert mag1._cached_array is not None

    with (ds_path / "save.p").open("wb") as f_write:
        pickle.dump(mag1, f_write)
    with (ds_path / "save.p").open("rb") as f_read:
        pickled_mag1 = pickle.load(f_read)

    # Make sure that the pickled mag can still read data
    assert pickled_mag1._cached_array is None
    np.testing.assert_array_equal(
        data_to_write,
        pickled_mag1.read(relative_offset=(0, 0, 0), size=data_to_write.shape[-3:]),
    )
    assert pickled_mag1._cached_array is not None

    # Make sure that the attributes of the MagView (not View) still exist
    assert pickled_mag1.layer is not None
