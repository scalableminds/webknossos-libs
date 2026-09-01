import numpy as np
import pytest
from upath import UPath

from tests.constants import (
    REMOTE_TESTOUTPUT_DIR,
    TESTDATA_DIR,
    TESTOUTPUT_DIR,
)
from tests.dataset._dataset_helpers import (
    DATA_FORMATS_AND_OUTPUT_PATHS,
    assure_exported_properties,
    copy_simple_dataset,
    default_chunk_config,
    prepare_dataset_path,
)
from webknossos.dataset import (
    Dataset,
)
from webknossos.dataset_properties import (
    COLOR_CATEGORY,
    SEGMENTATION_CATEGORY,
    DataFormat,
    LayerCategoryType,
)
from webknossos.geometry import (
    BoundingBox,
    Mag,
    Vec3Int,
)
from webknossos.utils import (
    copytree,
    rmtree,
)

rng = np.random.default_rng(1234)

pytestmark = pytest.mark.usefixtures("moto_server")


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_get_or_add_layer(data_format: DataFormat, output_path: UPath) -> None:
    ds_path = prepare_dataset_path(data_format, output_path)
    ds = Dataset(ds_path, voxel_size=(1, 1, 1))

    assert "color" not in ds.layers.keys()

    # layer did not exist before
    layer = ds.get_or_add_layer(
        "color",
        category=COLOR_CATEGORY,
        dtype="uint8",
        num_channels=1,
        data_format=data_format,
    )
    assert "color" in ds.layers.keys()
    assert layer.name == "color"
    assert layer.data_format == data_format

    # layer did exist before
    layer = ds.get_or_add_layer(
        "color",
        category=COLOR_CATEGORY,
        dtype="uint8",
        num_channels=1,
        data_format=data_format,
    )
    assert "color" in ds.layers.keys()
    assert layer.name == "color"
    assert layer.data_format == data_format

    with pytest.raises(AssertionError):
        # The layer "color" did exist before but with another dtype (this would work the same for 'category' and 'num_channels')
        ds.get_or_add_layer(
            "color",
            COLOR_CATEGORY,
            dtype="uint16",
            num_channels=1,
            data_format=data_format,
        )

    assure_exported_properties(ds)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_get_or_add_layer_idempotence(
    data_format: DataFormat, output_path: UPath
) -> None:
    ds_path = prepare_dataset_path(data_format, output_path)
    ds = Dataset(ds_path, voxel_size=(1, 1, 1))
    ds.get_or_add_layer(
        "color2", category="color", dtype=np.uint8, data_format=data_format
    ).get_or_add_mag("1")
    ds.get_or_add_layer(
        "color2", category="color", dtype=np.uint8, data_format=data_format
    ).get_or_add_mag("1")

    assure_exported_properties(ds)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_get_or_add_mag(data_format: DataFormat, output_path: UPath) -> None:
    ds_path = prepare_dataset_path(data_format, output_path)

    layer = Dataset(ds_path, voxel_size=(1, 1, 1)).add_layer(
        "color", category=COLOR_CATEGORY, data_format=data_format
    )

    assert Mag(1) not in layer.mags.keys()

    chunk_shape, shard_shape = default_chunk_config(data_format, 32)

    # The mag did not exist before
    mag = layer.get_or_add_mag(
        "1",
        chunk_shape=chunk_shape,
        shard_shape=shard_shape,
        compress=True,
    )
    assert Mag(1) in layer.mags.keys()
    assert mag.name == "1"
    assert mag.info.data_format == data_format

    # The mag did exist before
    layer.get_or_add_mag(
        "1",
        chunk_shape=chunk_shape,
        shard_shape=shard_shape,
        compress=True,
    )
    assert Mag(1) in layer.mags.keys()
    assert mag.name == "1"
    assert mag.info.data_format == data_format

    with pytest.raises(ValueError):
        # The mag "1" did exist before but with another 'chunk_shape' (this would work the same for 'shard_shape' and 'compress')
        layer.get_or_add_mag(
            "1",
            chunk_shape=Vec3Int.full(64),
            shard_shape=shard_shape,
            compress=True,
        )

    assure_exported_properties(layer.dataset)


def test_typing_of_get_mag() -> None:
    ds = Dataset.open(TESTDATA_DIR / "simple_wkw_dataset")
    layer = ds.get_layer("color")
    assert layer.get_mag("1") == layer.get_mag(1)
    assert layer.get_mag("1") == layer.get_mag((1, 1, 1))
    assert layer.get_mag("1") == layer.get_mag([1, 1, 1])
    assert layer.get_mag("1") == layer.get_mag(np.array([1, 1, 1]))
    assert layer.get_mag("1") == layer.get_mag(Mag(1))

    assure_exported_properties(ds)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_changing_layer_bounding_box(
    data_format: DataFormat, output_path: UPath
) -> None:
    ds_path = copy_simple_dataset(data_format, output_path, "change_layer_bounding_box")
    ds = Dataset.open(ds_path)
    layer = ds.get_layer("color")
    mag = layer.get_mag("1")

    bbox_size = ds.get_layer("color").bounding_box.size
    assert tuple(bbox_size) == (24, 24, 24)
    original_data = mag.read(absolute_offset=(0, 0, 0), size=bbox_size)
    assert original_data.shape == (3, 24, 24, 24)

    layer.bounding_box = layer.bounding_box.with_size(
        [
            12,
            12,
            10,
        ]
    )  # decrease bounding box

    bbox_size = ds.get_layer("color").bounding_box.size
    assert tuple(bbox_size) == (12, 12, 10)
    less_data = mag.read(absolute_offset=(0, 0, 0), size=bbox_size)
    assert less_data.shape == (3, 12, 12, 10)
    np.testing.assert_array_equal(original_data[:, :12, :12, :10], less_data)

    layer.bounding_box = layer.bounding_box.with_size(
        [
            36,
            48,
            60,
        ]
    )  # increase the bounding box

    bbox_size = ds.get_layer("color").bounding_box.size
    assert tuple(bbox_size) == (36, 48, 60)
    more_data = mag.read(absolute_offset=(0, 0, 0), size=bbox_size)
    assert more_data.shape == (3, 36, 48, 60)
    np.testing.assert_array_equal(more_data[:, :24, :24, :24], original_data)

    assert tuple(ds.get_layer("color").bounding_box.topleft) == (0, 0, 0)

    # Move the offset from (0, 0, 0) to (10, 10, 0)
    # Note that the bottom right coordinate of the dataset is still at (24, 24, 24)
    layer.bounding_box = BoundingBox((10, 10, 0), (14, 14, 24))

    new_bbox_offset = ds.get_layer("color").bounding_box.topleft
    new_bbox_size = ds.get_layer("color").bounding_box.size
    assert tuple(new_bbox_offset) == (10, 10, 0)
    assert tuple(new_bbox_size) == (14, 14, 24)
    np.testing.assert_array_equal(
        original_data,
        mag.read(absolute_offset=(0, 0, 0), size=mag.bounding_box.bottomright),
    )

    np.testing.assert_array_equal(
        original_data[:, 10:, 10:, :],
        mag.read(absolute_offset=(10, 10, 0), size=(14, 14, 24)),
    )

    # resetting the offset to (0, 0, 0)
    # Note that the size did not change. Therefore, the new bottom right is now at (14, 14, 24)
    layer.bounding_box = BoundingBox((0, 0, 0), new_bbox_size)
    new_data = mag.read()
    assert new_data.shape == (3, 14, 14, 24)
    np.testing.assert_array_equal(original_data[:, :14, :14, :], new_data)

    assure_exported_properties(ds)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_dataset_bounding_box_calculation(
    data_format: DataFormat, output_path: UPath
) -> None:
    ds_path = copy_simple_dataset(data_format, output_path, "change_layer_bounding_box")
    ds = Dataset.open(ds_path)
    layer = ds.get_layer("color")
    # BoundingBox(topleft=(0, 0, 0), size=(24, 24, 24))
    assert layer.bounding_box == ds.calculate_bounding_box(), (
        "The calculated bounding box of the dataset does not "
        + "match the color layer's bounding box."
    )
    layer.bounding_box = layer.bounding_box.with_size((512, 512, 512))
    assert layer.bounding_box == ds.calculate_bounding_box(), (
        "The calculated bounding box of the dataset does not "
        + "match the color layer's enlarged bounding box."
    )


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_search_dataset_also_for_long_layer_name(
    data_format: DataFormat, output_path: UPath
) -> None:
    ds_path = prepare_dataset_path(data_format, output_path, "long_layer_name")
    ds = Dataset(ds_path, voxel_size=(1, 1, 1))
    mag = ds.add_layer("color", COLOR_CATEGORY, data_format=data_format).add_mag("2")

    assert mag.name == "2"
    short_mag_file_path = ds.path / "color" / Mag(mag.name).to_layer_name()
    long_mag_file_path = ds.path / "color" / Mag(mag.name).to_long_layer_name()

    assert short_mag_file_path.exists()
    assert not long_mag_file_path.exists()

    write_data = rng.integers(0, 256, (10, 10, 10), dtype=np.uint8)
    mag.write(write_data, absolute_offset=(20, 20, 20), allow_resize=True)

    np.testing.assert_array_equal(
        mag.read(absolute_offset=(20, 20, 20), size=(20, 20, 20)),
        np.expand_dims(write_data, 0),
    )

    # rename the path from "long_layer_name/color/2" to "long_layer_name/color/2-2-2"
    copytree(short_mag_file_path, long_mag_file_path)
    rmtree(short_mag_file_path)

    # Remove path from mag to let the path be auto-detected
    ds._properties.data_layers[0].mags[0].path = None
    ds._save_dataset_properties()

    # make sure that reading data still works
    mag.read(absolute_offset=(20, 20, 20), size=(20, 20, 20))

    # when opening the dataset, it searches both for the long and the short path
    layer = Dataset.open(ds_path).get_layer("color")
    mag = layer.get_mag("2")
    np.testing.assert_array_equal(
        mag.read(absolute_offset=(20, 20, 20), size=(20, 20, 20)),
        np.expand_dims(write_data, 0),
    )
    layer.delete_mag("2")

    # Note: 'ds' is outdated (it still contains Mag(2)) because it was opened again and changed.
    assure_exported_properties(layer.dataset)


def test_get_or_add_layer_by_type() -> None:
    ds_path = prepare_dataset_path(DataFormat.WKW, TESTOUTPUT_DIR)
    ds = Dataset(ds_path, voxel_size=(1, 1, 1))
    assert len(ds.get_segmentation_layers()) == 0
    _ = ds.add_layer(
        "segmentation", SEGMENTATION_CATEGORY, largest_segment_id=999
    )  # adds layer
    assert len(ds.get_segmentation_layers()) == 1
    _ = ds.add_layer(
        "different_segmentation",
        SEGMENTATION_CATEGORY,
        largest_segment_id=999,
    )  # adds another layer
    assert len(ds.get_segmentation_layers()) == 2

    assert len(ds.get_color_layers()) == 0
    _ = ds.add_layer("color", COLOR_CATEGORY)  # adds layer
    assert len(ds.get_color_layers()) == 1
    _ = ds.add_layer("different_color", COLOR_CATEGORY)  # adds another layer
    assert len(ds.get_color_layers()) == 2

    assure_exported_properties(ds)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_rename_layer(data_format: DataFormat, output_path: UPath) -> None:
    ds_path = prepare_dataset_path(data_format, output_path)
    ds = Dataset(ds_path, voxel_size=(1, 1, 1))
    layer = ds.add_layer("color", COLOR_CATEGORY, data_format=data_format)
    mag = layer.add_mag(1)
    write_data = rng.integers(0, 256, (10, 20, 30), dtype=np.uint8)
    mag.write(data=write_data, allow_resize=True)

    if output_path == REMOTE_TESTOUTPUT_DIR:
        # Cannot rename layers on remote storage
        with pytest.raises(RuntimeError):
            layer.name = "color2"
        return
    else:
        layer.name = "color2"

    assert not (ds_path / "color").exists()
    assert (ds_path / "color2").exists()
    assert (
        len([layer for layer in ds._properties.data_layers if layer.name == "color"])
        == 0
    )
    assert (
        len([layer for layer in ds._properties.data_layers if layer.name == "color2"])
        == 1
    )
    assert ds._properties.data_layers[0].mags[0].path == "./color2/1"
    assert "color2" in ds.layers.keys()
    assert "color" not in ds.layers.keys()
    assert ds.get_layer("color2").data_format == data_format

    # The "mag" object which was created before renaming the layer is still valid
    np.testing.assert_array_equal(mag.read()[0], write_data)

    assure_exported_properties(ds)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_delete_layer_and_mag(data_format: DataFormat, output_path: UPath) -> None:
    ds_path = prepare_dataset_path(data_format, output_path)
    ds = Dataset(ds_path, voxel_size=(1, 1, 1))
    color_layer = ds.add_layer("color", COLOR_CATEGORY, data_format=data_format)
    color_layer.add_mag(1)
    color_layer.add_mag(2)
    ds.add_layer(
        "segmentation",
        SEGMENTATION_CATEGORY,
        largest_segment_id=999,
        data_format=data_format,
    )
    assert "color" in ds.layers
    assert "segmentation" in ds.layers
    assert (
        len([layer for layer in ds._properties.data_layers if layer.name == "color"])
        == 1
    )
    assert (
        len(
            [
                layer
                for layer in ds._properties.data_layers
                if layer.name == "segmentation"
            ]
        )
        == 1
    )
    assert len(color_layer._properties.mags) == 2

    color_layer.delete_mag(1)
    assert len(color_layer._properties.mags) == 1
    assert len([m for m in color_layer._properties.mags if Mag(m.mag) == Mag(2)]) == 1

    ds.delete_layer("color")
    assert "color" not in ds.layers
    assert "segmentation" in ds.layers
    assert (
        len([layer for layer in ds._properties.data_layers if layer.name == "color"])
        == 0
    )
    assert (
        len(
            [
                layer
                for layer in ds._properties.data_layers
                if layer.name == "segmentation"
            ]
        )
        == 1
    )

    assure_exported_properties(ds)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_add_layer_like(data_format: DataFormat, output_path: UPath) -> None:
    ds_path = prepare_dataset_path(data_format, output_path)
    ds = Dataset(ds_path, voxel_size=(1, 1, 1))
    color_layer1 = ds.add_layer(
        "color1",
        COLOR_CATEGORY,
        dtype="uint8",
        num_channels=3,
        data_format=data_format,
    )
    color_layer1.add_mag(1)
    segmentation_layer1 = ds.add_layer(
        "segmentation1",
        SEGMENTATION_CATEGORY,
        dtype="uint8",
        largest_segment_id=999,
        data_format=data_format,
    ).as_segmentation_layer()
    segmentation_layer1.add_mag(1)
    color_layer2 = ds.add_layer_like(color_layer1, "color2")
    segmentation_layer2 = ds.add_layer_like(
        segmentation_layer1, "segmentation2"
    ).as_segmentation_layer()

    assert color_layer1.name == "color1"
    assert color_layer2.name == "color2"
    assert len(color_layer1.mags) == 1
    assert len(color_layer2.mags) == 0
    assert color_layer1.category == color_layer2.category == COLOR_CATEGORY
    assert color_layer1.dtype == color_layer2.dtype == np.dtype("uint8")
    assert color_layer1.num_channels == color_layer2.num_channels == 3
    assert color_layer1.data_format == color_layer2.data_format == data_format

    assert segmentation_layer1.name == "segmentation1"
    assert segmentation_layer2.name == "segmentation2"
    assert len(segmentation_layer1.mags) == 1
    assert len(segmentation_layer2.mags) == 0
    assert (
        segmentation_layer1.category
        == segmentation_layer2.category
        == SEGMENTATION_CATEGORY
    )
    assert segmentation_layer1.dtype == segmentation_layer2.dtype == np.dtype("uint8")
    assert segmentation_layer1.num_channels == segmentation_layer2.num_channels == 1
    assert (
        segmentation_layer1.data_format
        == segmentation_layer2.data_format
        == data_format
    )
    assert (
        segmentation_layer1.largest_segment_id
        == segmentation_layer2.largest_segment_id
        == 999
    )

    assure_exported_properties(ds)


@pytest.mark.parametrize(
    "dtype,category,is_supported",
    [
        ("uint8", COLOR_CATEGORY, True),
        ("uint16", COLOR_CATEGORY, True),
        ("uint32", COLOR_CATEGORY, True),
        ("uint64", COLOR_CATEGORY, False),
        ("int8", COLOR_CATEGORY, True),
        ("int16", COLOR_CATEGORY, True),
        ("int32", COLOR_CATEGORY, True),
        ("int64", COLOR_CATEGORY, False),
        ("float32", COLOR_CATEGORY, True),
        ("float64", COLOR_CATEGORY, False),
        ("uint8", SEGMENTATION_CATEGORY, True),
        ("uint16", SEGMENTATION_CATEGORY, True),
        ("uint32", SEGMENTATION_CATEGORY, True),
        ("uint64", SEGMENTATION_CATEGORY, True),
        ("int8", SEGMENTATION_CATEGORY, True),
        ("int16", SEGMENTATION_CATEGORY, True),
        ("int32", SEGMENTATION_CATEGORY, True),
        ("int64", SEGMENTATION_CATEGORY, True),
        ("float32", SEGMENTATION_CATEGORY, False),
        ("float64", SEGMENTATION_CATEGORY, False),
    ],
)
def test_add_layer_dtype(
    dtype: str, category: LayerCategoryType, is_supported: bool
) -> None:
    ds_path = prepare_dataset_path(DataFormat.Zarr3, TESTOUTPUT_DIR, "dtype")
    ds = Dataset(ds_path, voxel_size=(1, 1, 1))
    if is_supported:
        layer = ds.add_layer(
            "test_layer",
            category=category,
            dtype=dtype,
        )
        assert layer.dtype == np.dtype(dtype)
    else:
        with pytest.raises(
            ValueError,
            match="Supported dtypes are:",
        ):
            ds.add_layer(
                "test_layer",
                category=category,
                dtype=dtype,
            )
