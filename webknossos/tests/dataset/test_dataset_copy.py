import json
from unittest import mock

import numpy as np
import pytest
from cluster_tools import get_executor
from upath import UPath

from tests.constants import (
    REMOTE_TESTOUTPUT_DIR,
    TESTDATA_DIR,
    TESTOUTPUT_DIR,
)
from tests.dataset._dataset_helpers import (
    DATA_FORMATS_AND_OUTPUT_PATHS,
    OUTPUT_PATHS,
    assure_exported_properties,
    copy_simple_dataset,
    prepare_dataset_path,
)
from webknossos.dataset import (
    AgglomerateAttachment,
    Dataset,
    MeshAttachment,
)
from webknossos.dataset.dataset import PROPERTIES_FILE_NAME
from webknossos.dataset.defaults import (
    DEFAULT_DATA_FORMAT,
)
from webknossos.dataset_properties import (
    COLOR_CATEGORY,
    SEGMENTATION_CATEGORY,
    AffineCoordinateTransformation,
    AttachmentDataFormat,
    DataFormat,
    DatasetViewConfiguration,
    LayerViewConfiguration,
)
from webknossos.geometry import (
    BoundingBox,
    Mag,
    Vec3Int,
)
from webknossos.utils import (
    dump_path,
    is_fs_path,
)

rng = np.random.default_rng(1234)

pytestmark = pytest.mark.usefixtures("moto_server")


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
@pytest.mark.parametrize("as_object", [True, False])
def test_add_layer_as_ref(
    data_format: DataFormat, output_path: UPath, as_object: bool
) -> None:
    ds_path = copy_simple_dataset(data_format, output_path, "original")
    new_path = prepare_dataset_path(data_format, output_path, "with_refs")

    # Add an additional segmentation layer to the original dataset
    original_ds = Dataset.open(ds_path)
    original_ds.add_layer(
        "segmentation", SEGMENTATION_CATEGORY, largest_segment_id=999
    ).add_mag(1)

    original_mag = original_ds.get_layer("color").get_mag("1")
    original_mag.write(
        rng.integers(0, 256, (3, 10, 10, 10), dtype=np.uint8), allow_unaligned=True
    )

    ds = Dataset(new_path, voxel_size=(1, 1, 1))
    # add color layer
    new_layer = ds.add_layer_as_ref(
        original_ds.get_layer("color") if as_object else ds_path / "color"
    )
    mag = new_layer.get_mag("1")
    # add segmentation layer
    new_segmentation_layer = ds.add_layer_as_ref(
        original_ds.get_layer("segmentation")
        if as_object
        else ds_path / "segmentation",
        new_layer_name="seg",
    )

    color_mag_path = original_mag.path.name
    assert ds._properties.data_layers[0].mags[0].path == dump_path(
        ds_path / "color" / color_mag_path, new_path
    )
    assert not (new_path / "color" / color_mag_path).exists()
    assert ds._properties.data_layers[1].mags[0].path == dump_path(
        ds_path / "segmentation" / "1", new_path
    )
    assert not (new_path / "segmentation" / "1").exists()
    assert not (new_path / "segmentation").exists()
    assert not (new_path / "seg" / "1").exists()
    assert not (new_path / "seg").exists()

    assert len(ds.layers) == 2
    assert len(ds.get_layer("color").mags) == 1

    assert new_segmentation_layer.as_segmentation_layer().largest_segment_id == 999

    assert not new_layer.read_only
    assert not new_segmentation_layer.read_only
    assert mag.read_only

    with pytest.raises(RuntimeError):
        mag.write(
            rng.integers(0, 256, (3, 10, 10, 10), dtype=np.uint8), allow_unaligned=True
        )

    np.testing.assert_array_equal(
        mag.read(absolute_offset=(0, 0, 0), size=(10, 10, 10)),
        original_mag.read(absolute_offset=(0, 0, 0), size=(10, 10, 10)),
    )

    assure_exported_properties(ds)


@pytest.mark.parametrize("output_path", OUTPUT_PATHS)
def test_add_layer_as_ref_prefix(output_path: UPath) -> None:
    source = Dataset(output_path / "name_with_suffix", (1, 1, 1))
    source.add_layer("consensus", SEGMENTATION_CATEGORY, dtype="uint8").add_mag(1)

    target = Dataset(output_path / "name", (1, 1, 1))
    target.add_layer("raw", COLOR_CATEGORY, dtype="uint8").add_mag(1)

    glom = source.get_layer("consensus")
    target.add_layer_as_ref(foreign_layer=glom, new_layer_name="glomeruli")

    assert target._properties.data_layers[1].mags[0].path == dump_path(
        source.get_layer("consensus").get_mag(1).path,
        UPath.home() / "random",  # an unrelated path
    )


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_ref_layer_add_mag(data_format: DataFormat, output_path: UPath) -> None:
    ds_path = copy_simple_dataset(data_format, output_path, "original")
    new_path = prepare_dataset_path(data_format, output_path, "with_refs")

    # Add an additional segmentation layer to the original dataset
    Dataset.open(ds_path).add_layer(
        "segmentation", SEGMENTATION_CATEGORY, largest_segment_id=999
    )

    ds = Dataset(new_path, voxel_size=(1, 1, 1))
    new_layer = ds.add_layer_as_ref(ds_path / "color")

    new_layer.add_mag(2)
    assert new_layer.get_mag(2).path == new_path / "color" / "2"
    assert new_layer.get_mag(2).path.exists()


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_ref_layer_rename(data_format: DataFormat, output_path: UPath) -> None:
    ds_path = copy_simple_dataset(data_format, output_path, "original")
    new_path = prepare_dataset_path(data_format, output_path, "with_ref")

    # Add an additional segmentation layer to the original dataset
    Dataset.open(ds_path).add_layer(
        "segmentation", SEGMENTATION_CATEGORY, largest_segment_id=999
    ).add_mag(1)

    ds = Dataset(new_path, voxel_size=(1, 1, 1))
    ref_layer = ds.add_layer_as_ref(ds_path / "color")

    assert not (new_path / "color").exists()
    if is_fs_path(new_path):
        ref_layer.name = "color2"

        with pytest.raises(ValueError):
            ref_layer.name = "color/2"  # invalid name
    else:
        with pytest.raises(RuntimeError):
            ref_layer.name = "color2"


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_add_mag_as_ref(data_format: DataFormat, output_path: UPath) -> None:
    ds_path = prepare_dataset_path(data_format, output_path, "original")
    new_path = prepare_dataset_path(data_format, output_path, "with_ref")

    original_ds = Dataset(ds_path, voxel_size=(1, 1, 1))
    original_layer = original_ds.add_layer(
        "color",
        COLOR_CATEGORY,
        dtype="uint8",
        bounding_box=BoundingBox((0, 0, 0), (10, 20, 30)),
    )
    original_layer.add_mag(1).write(
        data=rng.integers(0, 256, (10, 20, 30), dtype=np.uint8)
    )
    original_mag_2 = original_layer.add_mag(2)
    original_mag_2.write(data=rng.integers(0, 256, (5, 10, 15), dtype=np.uint8))
    original_mag_4 = original_layer.add_mag(4)
    original_mag_4.write(data=rng.integers(0, 256, (3, 5, 8), dtype=np.uint8))

    ds = Dataset(new_path, voxel_size=(1, 1, 1))
    layer = ds.add_layer(
        "color",
        COLOR_CATEGORY,
        dtype="uint8",
        bounding_box=BoundingBox((6, 6, 6), (10, 20, 30)),
    )
    layer.add_mag(1).write(
        absolute_offset=(6, 6, 6),
        data=rng.integers(0, 256, (10, 20, 30), dtype=np.uint8),
    )

    assert tuple(layer.bounding_box.topleft) == (6, 6, 6)
    assert tuple(layer.bounding_box.size) == (10, 20, 30)

    ref_mag_2 = layer.add_mag_as_ref(original_mag_2)
    ref_mag_4 = layer.add_mag_as_ref(ds_path / "color" / "4")
    assert ref_mag_2._properties.path == dump_path(ds_path / "color" / "2", new_path)
    assert ref_mag_4._properties.path == dump_path(ds_path / "color" / "4", new_path)

    assert (new_path / "color" / "1").exists()
    assert not (new_path / "color" / "2").exists()
    assert not (new_path / "color" / "4").exists()
    assert len(layer._properties.mags) == 3

    assert tuple(layer.bounding_box.topleft) == (0, 0, 0)
    assert tuple(layer.bounding_box.size) == (16, 26, 36)

    assert not layer.read_only
    assert not layer.get_mag(1).read_only
    assert ref_mag_2.read_only

    np.testing.assert_array_equal(
        ref_mag_2.read(absolute_offset=(0, 0, 0), size=(10, 10, 10))[0],
        original_layer.get_mag(2).read(absolute_offset=(0, 0, 0), size=(10, 10, 10))[0],
    )

    assure_exported_properties(ds)
    assure_exported_properties(original_ds)

    layer.delete_mag(4)
    assert Mag(4) not in layer.mags
    assert not (new_path / "color" / "4").exists()
    assert (ds_path / "color" / "4").exists()


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_add_mag_as_ref_with_mag(data_format: DataFormat, output_path: UPath) -> None:
    ds_path = prepare_dataset_path(data_format, output_path, "original")
    new_path = prepare_dataset_path(data_format, output_path, "with_ref")

    original_ds = Dataset(ds_path, voxel_size=(1, 1, 1))
    original_layer = original_ds.add_layer(
        "color",
        COLOR_CATEGORY,
        dtype="uint8",
        bounding_box=BoundingBox((0, 0, 0), (10, 20, 30)),
    )
    original_layer.add_mag(1).write(
        data=rng.integers(0, 256, (10, 20, 30), dtype=np.uint8)
    )

    ds = Dataset(new_path, voxel_size=(1, 1, 1))
    layer = ds.add_layer(
        "color",
        COLOR_CATEGORY,
        dtype="uint8",
        bounding_box=BoundingBox((6, 6, 6), (10, 20, 30)),
    )
    layer.add_mag_as_ref(original_layer.get_mag(1), mag="2")

    assert list(layer.mags.values())[0].mag == Mag("2")
    assert list(layer.mags.values())[0]._properties.path == dump_path(
        ds_path / "color" / "1", new_path
    )

    assure_exported_properties(ds)
    assure_exported_properties(original_ds)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_add_mag_as_copy(data_format: DataFormat, output_path: UPath) -> None:
    original_ds_path = prepare_dataset_path(data_format, output_path, "original")
    copy_ds_path = prepare_dataset_path(data_format, output_path, "copy")

    original_ds = Dataset(original_ds_path, voxel_size=(1, 1, 1))
    original_layer = original_ds.add_layer(
        "color",
        COLOR_CATEGORY,
        dtype="uint8",
        data_format=data_format,
        bounding_box=BoundingBox((6, 6, 6), (10, 20, 30)),
    )
    original_data = rng.integers(0, 256, (10, 20, 30), dtype=np.uint8)
    original_mag = original_layer.add_mag(1)
    original_mag.write(data=original_data, absolute_offset=(6, 6, 6))

    copy_ds = Dataset(copy_ds_path, voxel_size=(1, 1, 1))
    copy_layer = copy_ds.add_layer(
        "color", COLOR_CATEGORY, dtype="uint8", data_format=data_format
    )
    copy_mag = copy_layer.add_mag_as_copy(original_mag, extend_layer_bounding_box=True)
    assert not copy_mag.read_only

    assert (copy_ds_path / "color" / "1").exists()
    assert len(copy_layer._properties.mags) == 1

    assert tuple(copy_layer.bounding_box.topleft) == (6, 6, 6)
    assert tuple(copy_layer.bounding_box.size) == (10, 20, 30)

    # Write new data in copied layer
    new_data = rng.integers(0, 256, (5, 5, 5), dtype=np.uint8)
    copy_mag.write(
        absolute_offset=(0, 0, 0),
        data=new_data,
        allow_resize=True,
        allow_unaligned=True,
    )

    np.testing.assert_array_equal(
        copy_mag.read(absolute_offset=(0, 0, 0), size=(5, 5, 5))[0], new_data
    )
    np.testing.assert_array_equal(original_mag.read()[0], original_data)

    assure_exported_properties(original_ds)
    assure_exported_properties(copy_ds)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_add_fs_copy_mag(data_format: DataFormat, output_path: UPath) -> None:
    original_ds_path = prepare_dataset_path(data_format, output_path, "original")
    copy_ds_path = prepare_dataset_path(data_format, output_path, "copy")

    original_ds = Dataset(original_ds_path, voxel_size=(1, 1, 1))
    original_layer = original_ds.add_layer(
        "color",
        COLOR_CATEGORY,
        dtype="uint8",
        data_format=data_format,
        bounding_box=BoundingBox((6, 6, 6), (10, 20, 30)),
    )
    original_data = rng.integers(0, 256, (10, 20, 30), dtype=np.uint8)
    original_mag = original_layer.add_mag(1)
    original_mag.write(data=original_data, absolute_offset=(6, 6, 6))

    copy_ds = Dataset(copy_ds_path, voxel_size=(1, 1, 1))
    copy_layer = copy_ds.add_layer(
        "color", COLOR_CATEGORY, dtype="uint8", data_format=data_format
    )

    with mock.patch.object(
        copy_layer, "_add_fs_copy_mag", wraps=copy_layer._add_fs_copy_mag
    ) as mocked_method:
        copy_mag = copy_layer.add_mag_as_copy(
            original_mag, extend_layer_bounding_box=True
        )
        mocked_method.assert_called_once()

    assert not copy_layer.read_only
    assert not copy_mag.read_only

    assert (copy_ds_path / "color" / "1").exists()
    assert len(copy_layer._properties.mags) == 1

    assert tuple(copy_layer.bounding_box.topleft) == (6, 6, 6)
    assert tuple(copy_layer.bounding_box.size) == (10, 20, 30)

    # Write new data in copied layer
    new_data = rng.integers(0, 256, (5, 5, 5), dtype=np.uint8)
    copy_mag.write(
        absolute_offset=(0, 0, 0),
        data=new_data,
        allow_resize=True,
        allow_unaligned=True,
    )

    np.testing.assert_array_equal(
        copy_mag.read(absolute_offset=(0, 0, 0), size=(5, 5, 5))[0], new_data
    )
    np.testing.assert_array_equal(original_mag.read()[0], original_data)

    assure_exported_properties(original_ds)
    assure_exported_properties(copy_ds)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_dataset_shallow_copy(data_format: DataFormat, output_path: UPath) -> None:
    ds_path = prepare_dataset_path(data_format, output_path, "original")
    copy_path = prepare_dataset_path(data_format, output_path, "copy")

    ds = Dataset(ds_path, (1, 1, 1))
    ds.default_view_configuration = DatasetViewConfiguration(zoom=1.5)
    original_layer_1 = ds.add_layer(
        "color",
        COLOR_CATEGORY,
        dtype=np.uint8,
        num_channels=1,
        data_format=data_format,
    )
    original_layer_1.add_mag(1)
    original_layer_1.add_mag("2-2-1")
    original_layer_2 = ds.add_layer(
        "segmentation",
        SEGMENTATION_CATEGORY,
        dtype=np.uint32,
        largest_segment_id=0,
        data_format=data_format,
    ).as_segmentation_layer()
    original_layer_2.add_mag(4)
    agglomerates_path = original_layer_2.path / "agglomerates" / "agglomerate_view.hdf5"
    agglomerates_path.parent.mkdir(parents=True)
    agglomerates_path.touch()
    original_layer_2.attachments.add_attachment_as_ref(
        AgglomerateAttachment.from_path_and_name(
            agglomerates_path,
            name="agglomerate_view",
            data_format=AttachmentDataFormat.HDF5,
        )
    )

    shallow_copy_of_ds = ds.shallow_copy_dataset(copy_path)
    assert (
        shallow_copy_of_ds.default_view_configuration
        and shallow_copy_of_ds.default_view_configuration.zoom == 1.5
    )
    shallow_copy_of_ds.get_layer("color").add_mag(Mag("4-4-1"))
    assert len(Dataset.open(ds_path).get_layer("color").mags) == 2, (
        "Adding a new mag should not affect the original dataset"
    )
    assert len(Dataset.open(copy_path).get_layer("color").mags) == 3, (
        "Expecting all mags from original dataset and new downsampled mag"
    )
    assert str(
        shallow_copy_of_ds.get_segmentation_layer("segmentation")
        .attachments.agglomerates[0]
        .path
    ) == str(ds_path / "segmentation" / "agglomerates" / "agglomerate_view.hdf5"), (
        "Expecting agglomerates to exist in shallow copy"
    )

    assert not (
        copy_path / "segmentation" / "agglomerates" / "agglomerate_view.hdf5"
    ).exists(), "Expecting agglomerates not to exist in shallow copy"

    assert not shallow_copy_of_ds.get_layer("color").read_only
    assert shallow_copy_of_ds.get_layer("color").get_mag(1).read_only


def test_dataset_shallow_copy_downsample() -> None:
    ds_path = prepare_dataset_path(DEFAULT_DATA_FORMAT, TESTOUTPUT_DIR, "original")
    copy_path = prepare_dataset_path(DEFAULT_DATA_FORMAT, TESTOUTPUT_DIR, "copy")

    ds = Dataset(ds_path, (1, 1, 1))
    original_layer_1 = ds.add_layer(
        "color",
        COLOR_CATEGORY,
        dtype=np.uint8,
        num_channels=1,
        data_format=DEFAULT_DATA_FORMAT,
        bounding_box=BoundingBox((0, 0, 0), (512, 512, 512)),
    )
    original_layer_1.add_mag(1)

    # Creating a shallow copy
    shallow_copy_of_ds = ds.shallow_copy_dataset(copy_path)
    # Pre-initializing the downsampled mags
    shallow_copy_of_ds.get_layer("color").downsample(
        from_mag=Mag(1), coarsest_mag=Mag(2), only_setup_mags=True
    )
    # Re-opening the copy dataset in order to re-determine read-only mags
    shallow_copy_of_ds = Dataset.open(copy_path)
    with get_executor("sequential") as ex:
        shallow_copy_of_ds.get_layer("color").downsample(
            from_mag=Mag(1), coarsest_mag=Mag(2), allow_overwrite=True, executor=ex
        )

    assert not shallow_copy_of_ds.get_layer("color").read_only
    assert shallow_copy_of_ds.get_layer("color").get_mag(1).read_only


def test_dataset_conversion_wkw_only() -> None:
    ds_path = prepare_dataset_path(DataFormat.WKW, TESTOUTPUT_DIR, "original")
    converted_path = prepare_dataset_path(DataFormat.WKW, TESTOUTPUT_DIR, "converted")

    # create example dataset
    origin_ds = Dataset(ds_path, voxel_size=(1, 1, 1))
    origin_ds.default_view_configuration = DatasetViewConfiguration(zoom=1.5)
    seg_layer = origin_ds.add_layer(
        "layer1",
        SEGMENTATION_CATEGORY,
        num_channels=1,
        largest_segment_id=1000000000,
    )
    seg_layer.add_mag(
        "1", chunk_shape=Vec3Int.full(8), shard_shape=Vec3Int.full(128)
    ).write(
        absolute_offset=(10, 20, 30),
        data=rng.integers(0, 256, (128, 128, 256), dtype=np.uint8),
        allow_resize=True,
    )
    seg_layer.add_mag(
        "2", chunk_shape=Vec3Int.full(8), shard_shape=Vec3Int.full(128)
    ).write(
        absolute_offset=(10, 20, 30),
        data=rng.integers(0, 256, (64, 64, 128), dtype=np.uint8),
        allow_resize=True,
    )
    wk_color_layer = origin_ds.add_layer("layer2", COLOR_CATEGORY, num_channels=3)
    wk_color_layer.add_mag(
        "1", chunk_shape=Vec3Int.full(8), shard_shape=Vec3Int.full(128)
    ).write(
        absolute_offset=(10, 20, 30),
        data=rng.integers(0, 256, (3, 128, 128, 256), dtype=np.uint8),
        allow_resize=True,
    )
    wk_color_layer.add_mag(
        "2", chunk_shape=Vec3Int.full(8), shard_shape=Vec3Int.full(128)
    ).write(
        absolute_offset=(10, 20, 30),
        data=rng.integers(0, 256, (3, 64, 64, 128), dtype=np.uint8),
        allow_resize=True,
    )
    converted_ds = origin_ds.copy_dataset(converted_path)

    assert (
        converted_ds.default_view_configuration
        and converted_ds.default_view_configuration.zoom == 1.5
    )
    assert origin_ds.layers.keys() == converted_ds.layers.keys()
    for layer_name in origin_ds.layers:
        assert (
            origin_ds.layers[layer_name].mags.keys()
            == converted_ds.layers[layer_name].mags.keys()
        )
        for mag in origin_ds.layers[layer_name].mags:
            origin_info = origin_ds.layers[layer_name].mags[mag].info
            converted_info = converted_ds.layers[layer_name].mags[mag].info
            assert origin_info.voxel_type == converted_info.voxel_type
            assert origin_info.bounding_box.size.c == converted_info.bounding_box.size.c
            assert origin_info.compression_mode == converted_info.compression_mode
            assert origin_info.chunk_shape == converted_info.chunk_shape
            assert origin_info.data_format == converted_info.data_format
            np.testing.assert_array_equal(
                origin_ds.layers[layer_name].mags[mag].read(),
                converted_ds.layers[layer_name].mags[mag].read(),
            )

    assure_exported_properties(origin_ds)
    assure_exported_properties(converted_ds)


@pytest.mark.parametrize("output_path", [TESTOUTPUT_DIR, REMOTE_TESTOUTPUT_DIR])
@pytest.mark.parametrize("data_format", [DataFormat.Zarr, DataFormat.Zarr3])
def test_dataset_conversion_from_wkw_to_zarr(
    output_path: UPath, data_format: DataFormat
) -> None:
    converted_path = prepare_dataset_path(data_format, output_path, "converted")

    input_ds = Dataset.open(TESTDATA_DIR / "simple_wkw_dataset")
    print(input_ds.get_layer("color").get_mag("1").info.chunk_shape)
    converted_ds = input_ds.copy_dataset(
        converted_path,
        data_format=data_format,
        shard_shape=8 if data_format == DataFormat.Zarr else 32,
    )

    if data_format == DataFormat.Zarr:
        assert (converted_path / "color" / "1" / ".zarray").exists()
    else:
        assert (converted_path / "color" / "1" / "zarr.json").exists()
    assert np.all(
        input_ds.get_layer("color").get_mag("1").read()
        == converted_ds.get_layer("color").get_mag("1").read()
    )
    assert converted_ds.get_layer("color").data_format == data_format
    assert converted_ds.get_layer("color").get_mag("1").info.data_format == data_format

    assure_exported_properties(converted_ds)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_add_layer_as_copy(data_format: DataFormat, output_path: UPath) -> None:
    ds_path = prepare_dataset_path(data_format, output_path, "original")
    copy_path = prepare_dataset_path(data_format, output_path, "copy")

    ds = Dataset(ds_path, voxel_size=(2, 2, 1))

    # Create dataset to copy data from
    other_ds = Dataset(copy_path, voxel_size=(2, 2, 1))
    original_color_layer = other_ds.add_layer(
        "color", COLOR_CATEGORY, data_format=data_format
    )
    original_color_layer.add_mag(1).write(
        absolute_offset=(10, 20, 30),
        data=rng.integers(0, 256, (32, 64, 128), dtype=np.uint8),
        allow_resize=True,
    )
    other_ds.add_layer(
        "segmentation",
        SEGMENTATION_CATEGORY,
        data_format=data_format,
        largest_segment_id=999,
    ).add_mag("1")

    # Copies the "color" layer from a different dataset
    ds.add_layer_as_copy(copy_path / "color")
    ds.add_layer_as_copy(copy_path / "segmentation")
    assert len(ds.layers) == 2
    assert (
        ds.get_layer("segmentation").as_segmentation_layer().largest_segment_id == 999
    )

    color_layer = ds.get_layer("color")
    assert color_layer.bounding_box == BoundingBox(
        topleft=(10, 20, 30), size=(32, 64, 128)
    )
    assert color_layer.mags.keys() == original_color_layer.mags.keys()
    assert len(color_layer.mags.keys()) >= 1
    for mag in color_layer.mags.keys():
        np.testing.assert_array_equal(
            color_layer.get_mag(mag).read(), original_color_layer.get_mag(mag).read()
        )
        # Test if the copied layer contains actual data
        assert np.max(color_layer.get_mag(mag).read()) > 0

    with pytest.raises(IndexError):
        # The dataset already has a layer called "color".
        ds.add_layer_as_copy(copy_path / "color")

    # Test if the changes of the properties are persisted on disk by opening it again
    assert "color" in Dataset.open(ds_path).layers.keys()

    assure_exported_properties(ds)


def test_copy_preserves_layer_metadata() -> None:
    """Copying a layer or a whole dataset carries over the optional layer metadata."""
    ds_path = prepare_dataset_path(DataFormat.WKW, TESTOUTPUT_DIR, "metadata_original")
    layer_copy_path = prepare_dataset_path(
        DataFormat.WKW, TESTOUTPUT_DIR, "metadata_layer_copy"
    )
    dataset_copy_path = prepare_dataset_path(
        DataFormat.WKW, TESTOUTPUT_DIR, "metadata_dataset_copy"
    )

    ds = Dataset(ds_path, voxel_size=(1, 1, 1))
    layer = ds.add_layer("color", COLOR_CATEGORY)
    layer.add_mag(1)
    view_configuration = LayerViewConfiguration(color=(255, 0, 0), alpha=50.0)
    coordinate_transformations = (
        AffineCoordinateTransformation.from_translation((1, 2, 3)),
    )
    layer.default_view_configuration = view_configuration
    layer.coordinate_transformations = coordinate_transformations

    # add_layer_as_copy
    copied_layer = Dataset(layer_copy_path, voxel_size=(1, 1, 1)).add_layer_as_copy(
        layer, "color_copy"
    )
    assert copied_layer.default_view_configuration == view_configuration
    assert copied_layer.coordinate_transformations == coordinate_transformations

    # copy_dataset
    copied_dataset_layer = ds.copy_dataset(dataset_copy_path).get_layer("color")
    assert copied_dataset_layer.default_view_configuration == view_configuration
    assert copied_dataset_layer.coordinate_transformations == coordinate_transformations

    # The copies must not share any mutable state with the layer they were copied from.
    # `view_configuration` is the very object that was assigned, so it is mutated too
    # and cannot serve as the expected value here.
    expected = LayerViewConfiguration(color=(255, 0, 0), alpha=50.0)
    assert layer.default_view_configuration is not None
    layer.default_view_configuration.color = (0, 255, 0)
    assert copied_layer.default_view_configuration == expected
    assert copied_dataset_layer.default_view_configuration == expected

    # The same holds for a layer that was added with `add_layer_like`, which must not
    # end up sharing the stored list of transformations with the layer it was created
    # from
    like_layer = Dataset(
        prepare_dataset_path(DataFormat.WKW, TESTOUTPUT_DIR, "metadata_like"),
        voxel_size=(1, 1, 1),
    ).add_layer_like(layer, "color_like")
    assert like_layer.coordinate_transformations == coordinate_transformations
    assert (
        like_layer._properties.coordinate_transformations
        is not layer._properties.coordinate_transformations
    )
    like_layer.coordinate_transformations = []
    assert layer.coordinate_transformations == coordinate_transformations

    # Copying onto an existing layer keeps the metadata that the source does not have
    existing_dataset = Dataset(
        prepare_dataset_path(DataFormat.WKW, TESTOUTPUT_DIR, "metadata_existing"),
        voxel_size=(1, 1, 1),
    )
    existing_layer = existing_dataset.add_layer("color", COLOR_CATEGORY)
    existing_view_configuration = LayerViewConfiguration(alpha=42.0)
    existing_layer.default_view_configuration = existing_view_configuration
    bare_source = Dataset(
        prepare_dataset_path(DataFormat.WKW, TESTOUTPUT_DIR, "metadata_bare_source"),
        voxel_size=(1, 1, 1),
    ).add_layer("color", COLOR_CATEGORY)
    bare_source.add_mag(1)
    existing_dataset.add_layer_as_copy(bare_source, "color", exists_ok=True)
    assert existing_layer.default_view_configuration == existing_view_configuration

    # A layer without this metadata does not get empty entries in the properties
    layer.default_view_configuration = None
    layer.coordinate_transformations = []
    bare_dataset = Dataset(
        prepare_dataset_path(DataFormat.WKW, TESTOUTPUT_DIR, "metadata_bare"),
        voxel_size=(1, 1, 1),
    )
    bare_layer = bare_dataset.add_layer_as_copy(layer, "color_bare")
    assert bare_layer.default_view_configuration is None
    assert bare_layer.coordinate_transformations == ()
    properties = json.loads((bare_dataset.path / PROPERTIES_FILE_NAME).read_text())
    assert "defaultViewConfiguration" not in properties["dataLayers"][0]
    assert "coordinateTransformations" not in properties["dataLayers"][0]

    assure_exported_properties(bare_dataset)


@pytest.mark.parametrize("data_format", [DataFormat.Zarr, DataFormat.Zarr3])
def test_zarr_copy_to_remote_dataset(data_format: DataFormat) -> None:
    ds_path = prepare_dataset_path(data_format, REMOTE_TESTOUTPUT_DIR, "copied")
    Dataset.open(TESTDATA_DIR / "simple_zarr_dataset").copy_dataset(
        ds_path,
        shard_shape=32,
        data_format=data_format,
    )
    if data_format == DataFormat.Zarr:
        assert (ds_path / "color" / "1" / ".zarray").exists()
    else:
        assert (ds_path / "color" / "1" / "zarr.json").exists()


@pytest.mark.parametrize("input_path", OUTPUT_PATHS)
@pytest.mark.parametrize("output_path", OUTPUT_PATHS)
def test_copy_dataset_with_attachments(input_path: UPath, output_path: UPath) -> None:
    ds_path = copy_simple_dataset(DEFAULT_DATA_FORMAT, input_path)
    new_ds_path = prepare_dataset_path(DEFAULT_DATA_FORMAT, output_path, "copied")

    ds = Dataset.open(ds_path)
    ds.default_view_configuration = DatasetViewConfiguration(zoom=1.5)
    # Add segmentation layer and meshfile
    seg_layer = ds.add_layer(
        "segmentation",
        SEGMENTATION_CATEGORY,
        largest_segment_id=999,
        bounding_box=BoundingBox((0, 0, 0), (10, 10, 10)),
    ).as_segmentation_layer()
    seg_mag = seg_layer.add_mag(1)
    seg_mag.write(data=np.zeros((10, 10, 10), dtype=np.uint8))

    meshfile_path = seg_layer.path / "meshes" / "meshfile"
    meshfile_path.mkdir(parents=True, exist_ok=True)
    (meshfile_path / "zarr.json").write_text("test")

    seg_layer.attachments.add_attachment_as_ref(
        MeshAttachment.from_path_and_name(
            meshfile_path, "meshfile", data_format=AttachmentDataFormat.Zarr3
        )
    )

    # Copy
    copy_ds = ds.copy_dataset(new_ds_path)

    assert (
        copy_ds.default_view_configuration
        and copy_ds.default_view_configuration.zoom == 1.5
    )
    assert (new_ds_path / "segmentation" / "1" / "zarr.json").exists()
    assert (new_ds_path / "segmentation" / "meshes" / "meshfile" / "zarr.json").exists()


def test_wkw_copy_to_remote_dataset() -> None:
    ds_path = prepare_dataset_path(DataFormat.WKW, REMOTE_TESTOUTPUT_DIR, "copied")
    wkw_ds = Dataset.open(TESTDATA_DIR / "simple_wkw_dataset")

    # Fails with explicit data_format=wkw ...
    with pytest.warns(UserWarning, match=".*not recommended.*"):
        wkw_ds.copy_dataset(ds_path, shard_shape=32, data_format=DataFormat.WKW)

    # ... and with implicit data_format=wkw from the source layers.
    ds_path = prepare_dataset_path(DataFormat.WKW, REMOTE_TESTOUTPUT_DIR, "copied2")
    with pytest.warns(UserWarning, match=".*not recommended.*"):
        wkw_ds.copy_dataset(
            ds_path,
            shard_shape=32,
        )


def test_copy_dataset_exists_ok() -> None:
    ds_path = prepare_dataset_path(DataFormat.WKW, REMOTE_TESTOUTPUT_DIR, "copied")
    wkw_ds = Dataset.open(TESTDATA_DIR / "simple_wkw_dataset")

    wkw_ds.copy_dataset(ds_path, data_format=DataFormat.Zarr3)
    with pytest.raises(RuntimeError):
        wkw_ds.copy_dataset(ds_path, data_format=DataFormat.Zarr3)
    wkw_ds.copy_dataset(ds_path, data_format=DataFormat.Zarr3, exists_ok=True)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_add_layer_as_copy_exists_ok(
    data_format: DataFormat, output_path: UPath
) -> None:
    ds_path = prepare_dataset_path(data_format, output_path, "copy_exists_ok")
    source_path = prepare_dataset_path(data_format, output_path, "copy_exists_ok_src")

    ds = Dataset(ds_path, voxel_size=(2, 2, 1))

    # Create source dataset with a color layer
    source_ds = Dataset(source_path, voxel_size=(2, 2, 1))
    source_layer = source_ds.add_layer("color", COLOR_CATEGORY, data_format=data_format)
    source_layer.add_mag(1).write(
        absolute_offset=(0, 0, 0),
        data=rng.integers(0, 256, (16, 16, 16), dtype=np.uint8),
        allow_resize=True,
    )

    # First copy should succeed
    ds.add_layer_as_copy(source_layer)
    assert "color" in ds.layers

    # Second copy without exists_ok should fail
    with pytest.raises(IndexError):
        ds.add_layer_as_copy(source_layer)

    assure_exported_properties(ds)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_add_layer_as_copy_with_rename(
    data_format: DataFormat, output_path: UPath
) -> None:
    ds_path = prepare_dataset_path(data_format, output_path, "copy_rename")
    source_path = prepare_dataset_path(data_format, output_path, "copy_rename_src")

    ds = Dataset(ds_path, voxel_size=(2, 2, 1))

    # Create source dataset
    source_ds = Dataset(source_path, voxel_size=(2, 2, 1))
    source_layer = source_ds.add_layer("color", COLOR_CATEGORY, data_format=data_format)
    write_data = rng.integers(0, 256, (16, 16, 16), dtype=np.uint8)
    source_layer.add_mag(1).write(
        absolute_offset=(0, 0, 0),
        data=write_data,
        allow_resize=True,
    )

    # Copy with a different name
    ds.add_layer_as_copy(source_layer, new_layer_name="color_copy")
    assert "color_copy" in ds.layers
    assert "color" not in ds.layers

    np.testing.assert_array_equal(
        ds.get_layer("color_copy").get_mag(1).read(),
        source_layer.get_mag(1).read(),
    )

    assure_exported_properties(ds)


def test_add_mag_ref_from_local_path(tmp_upath: UPath) -> None:
    dataset1 = Dataset(tmp_upath / "origin", voxel_size=(10, 10, 10))
    dataset1.write_layer(
        "color",
        COLOR_CATEGORY,
        data=np.ones((1, 10, 10, 10), dtype="uint8"),
        downsample=False,
    )

    dataset2 = Dataset(tmp_upath / "copy", voxel_size=(10, 10, 10))
    layer1 = dataset2.add_layer_as_ref(tmp_upath / "origin" / "color")
    layer1_mag1 = layer1.get_mag(1)

    assert layer1_mag1.path == tmp_upath / "origin" / "color" / "1"
    assert (
        layer1_mag1._properties.path
        == (tmp_upath / "origin" / "color" / "1").resolve().as_posix()
    )

    layer2_mag1 = dataset2.add_layer("color2", COLOR_CATEGORY).add_mag_as_ref(
        tmp_upath / "origin" / "color" / "1"
    )
    assert layer2_mag1.path == tmp_upath / "origin" / "color" / "1"
    assert (
        layer2_mag1._properties.path
        == (tmp_upath / "origin" / "color" / "1").resolve().as_posix()
    )
