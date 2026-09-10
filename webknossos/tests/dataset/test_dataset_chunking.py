import itertools
import json

import numpy as np
import pytest
from cluster_tools import get_executor
from upath import UPath

from tests.constants import (
    REMOTE_TESTOUTPUT_DIR,
    TESTOUTPUT_DIR,
)
from tests.dataset._dataset_helpers import (
    DATA_FORMATS,
    DATA_FORMATS_AND_OUTPUT_PATHS,
    OUTPUT_PATHS,
    assure_exported_properties,
    chunk_job,
    copy_and_transform_job,
    default_chunk_config,
    for_each_chunking_advanced,
    for_each_chunking_with_wrong_chunk_shape,
    prepare_dataset_path,
)
from webknossos.dataset import (
    Dataset,
    View,
)
from webknossos.dataset.defaults import (
    DEFAULT_CHUNK_SHAPE,
    DEFAULT_DATA_FORMAT,
    DEFAULT_SHARD_SHAPE,
)
from webknossos.dataset.layer.view._array import Zarr3ArrayInfo, Zarr3Config
from webknossos.dataset_properties import (
    COLOR_CATEGORY,
    DataFormat,
)
from webknossos.geometry import (
    BoundingBox,
    Vec3Int,
)
from webknossos.utils import (
    is_remote_path,
    named_partial,
)

rng = np.random.default_rng(1234)

pytestmark = pytest.mark.usefixtures("moto_server")


def test_chunked_compressed_write() -> None:
    ds_path = prepare_dataset_path(DataFormat.WKW, TESTOUTPUT_DIR)
    mag = (
        Dataset(ds_path, voxel_size=(1, 1, 1))
        .get_or_add_layer(
            "color",
            COLOR_CATEGORY,
            data_format=DataFormat.WKW,
            bounding_box=BoundingBox(
                DEFAULT_SHARD_SHAPE - Vec3Int(5, 5, 5), Vec3Int(10, 10, 10)
            ),
        )
        .get_or_add_mag(
            "1",
            compress=True,
        )
    )

    data: np.ndarray = rng.integers(0, 256, (10, 10, 10), dtype=np.uint8)

    # write data in the bottom-right cornor of a shard so that other shards have to be written too
    mag.write(data, absolute_offset=mag.info.shard_shape - Vec3Int(5, 5, 5))

    assert (
        mag.get_view(
            absolute_offset=mag.info.shard_shape - Vec3Int(5, 5, 5),
            size=Vec3Int(10, 10, 10),
        ).read()
        == data
    ).all()


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_chunking_wk(data_format: DataFormat, output_path: UPath) -> None:
    ds_path = prepare_dataset_path(data_format, output_path)
    ds = Dataset(ds_path, voxel_size=(2, 2, 1))
    chunk_shape, shard_shape = default_chunk_config(data_format, 8)

    layer = ds.add_layer("color", COLOR_CATEGORY, data_format=data_format)
    mag = layer.add_mag(
        "1",
        shard_shape=shard_shape,
        chunk_shape=chunk_shape,
    )

    original_data = rng.integers(0, 206, (50, 100, 150), dtype=np.uint8)
    mag.write(absolute_offset=(70, 80, 90), data=original_data, allow_resize=True)

    # Test with executor
    with get_executor("sequential") as executor:
        mag.for_each_chunk(
            chunk_job,
            chunk_shape=shard_shape,
            executor=executor,
        )
    np.testing.assert_array_equal(original_data + 50, mag.get_view().read()[0])

    # Reset the data
    mag.write(absolute_offset=(70, 80, 90), data=original_data, allow_resize=True)

    # Test without executor
    mag.for_each_chunk(
        chunk_job,
        chunk_shape=shard_shape,
    )
    np.testing.assert_array_equal(original_data + 50, mag.get_view().read()[0])

    assure_exported_properties(ds)


@pytest.mark.parametrize("data_format", [DataFormat.WKW, DataFormat.Zarr3])
def test_chunking_wkw_advanced(data_format: DataFormat) -> None:
    ds_path = prepare_dataset_path(data_format, TESTOUTPUT_DIR, "chunking_advanced")
    ds = Dataset(ds_path, voxel_size=(1, 1, 2))

    mag = ds.add_layer(
        "color",
        category=COLOR_CATEGORY,
        dtype="uint8",
        num_channels=3,
    ).add_mag(
        "1",
        chunk_shape=8,
        shard_shape=64,
    )
    mag.write(
        data=rng.integers(0, 256, (3, 256, 256, 256), dtype=np.uint8),
        allow_resize=True,
    )
    with pytest.warns(UserWarning, match=".*not aligned with the shard shape.*"):
        view = mag.get_view(absolute_offset=(10, 10, 10), size=(150, 150, 54))
        for_each_chunking_advanced(ds, view)

    assure_exported_properties(ds)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_chunking_wkw_wrong_chunk_shape(
    data_format: DataFormat, output_path: UPath
) -> None:
    ds_path = prepare_dataset_path(
        data_format, output_path, "chunking_with_wrong_chunk_shape"
    )
    ds = Dataset(ds_path, voxel_size=(1, 1, 2))
    chunk_shape, shard_shape = default_chunk_config(data_format, 8)
    mag = ds.add_layer(
        "color",
        category=COLOR_CATEGORY,
        dtype="uint8",
        num_channels=3,
        data_format=data_format,
    ).add_mag(
        "1",
        chunk_shape=chunk_shape,
        shard_shape=shard_shape,
    )
    mag.write(
        data=rng.integers(0, 256, (3, 256, 256, 256), dtype=np.uint8),
        allow_resize=True,
    )
    view = mag.get_view()

    for_each_chunking_with_wrong_chunk_shape(view)

    assure_exported_properties(ds)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_writing_subset_of_compressed_data_multi_channel(
    data_format: DataFormat, output_path: UPath
) -> None:
    ds_path = prepare_dataset_path(data_format, output_path, "compressed_data")
    chunk_shape, shard_shape = default_chunk_config(data_format, 8)

    # create uncompressed dataset
    write_data1 = rng.integers(0, 256, (3, 100, 120, 140), dtype=np.uint8)
    mag_view = (
        Dataset(ds_path, voxel_size=(1, 1, 1))
        .add_layer("color", COLOR_CATEGORY, num_channels=3, data_format=data_format)
        .add_mag(
            "1",
            chunk_shape=chunk_shape,
            shard_shape=shard_shape,
            compress=True,
        )
    )
    mag_view.write(write_data1, allow_resize=True, allow_unaligned=True)

    # open compressed dataset
    compressed_mag = Dataset.open(ds_path).get_layer("color").get_mag("1")

    write_data2 = rng.integers(0, 256, (3, 10, 10, 10), dtype=np.uint8)
    # Writing unaligned data to a compressed dataset works because the data gets
    # padded, but it requires an explicit allow_unaligned=True flag
    # Writing compressed data directly to "compressed_mag" also works, but using a
    # View here covers an additional edge case
    with pytest.warns(UserWarning):
        view = compressed_mag.get_view(relative_offset=(50, 60, 70), size=(50, 60, 70))
    with pytest.raises(ValueError):
        view.write(relative_offset=(10, 20, 30), data=write_data2)
    view.write(relative_offset=(10, 20, 30), data=write_data2, allow_unaligned=True)

    np.testing.assert_array_equal(
        write_data2,
        compressed_mag.read(relative_offset=(60, 80, 100), size=(10, 10, 10)),
    )  # the new data was written
    np.testing.assert_array_equal(
        write_data1[:, :60, :80, :100],
        compressed_mag.read(relative_offset=(0, 0, 0), size=(60, 80, 100)),
    )  # the old data is still there


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_writing_subset_of_compressed_data_single_channel(
    data_format: DataFormat, output_path: UPath
) -> None:
    ds_path = prepare_dataset_path(data_format, output_path, "compressed_data")
    chunk_shape, shard_shape = default_chunk_config(data_format, 8)

    # create uncompressed dataset
    write_data1 = rng.integers(0, 256, (100, 120, 140), dtype=np.uint8)
    mag_view = (
        Dataset(ds_path, voxel_size=(1, 1, 1))
        .add_layer("color", COLOR_CATEGORY, data_format=data_format)
        .add_mag(
            "1",
            chunk_shape=chunk_shape,
            shard_shape=shard_shape,
            compress=True,
        )
    )
    mag_view.write(write_data1, allow_resize=True)

    # open compressed dataset
    compressed_mag = Dataset.open(ds_path).get_layer("color").get_mag("1")

    write_data2 = rng.integers(0, 256, (10, 10, 10), dtype=np.uint8)

    # Writing unaligned data to a compressed dataset works because the data gets
    # padded, but it requires an explicit allow_unaligned=True flag
    # Writing compressed data directly to "compressed_mag" also works, but using a
    # View here covers an additional edge case
    with pytest.warns(UserWarning):
        view = compressed_mag.get_view(absolute_offset=(50, 60, 70), size=(50, 60, 70))
    with pytest.raises(ValueError, match=".*not aligned with the shard shape.*"):
        view.write(relative_offset=(10, 20, 30), data=write_data2)
    view.write(relative_offset=(10, 20, 30), data=write_data2, allow_unaligned=True)

    np.testing.assert_array_equal(
        write_data2,
        compressed_mag.read(absolute_offset=(60, 80, 100), size=(10, 10, 10))[0],
    )  # the new data was written
    np.testing.assert_array_equal(
        write_data1[:60, :80, :100],
        compressed_mag.read(absolute_offset=(0, 0, 0), size=(60, 80, 100))[0],
    )  # the old data is still there


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_writing_subset_of_compressed_data(
    data_format: DataFormat, output_path: UPath
) -> None:
    ds_path = prepare_dataset_path(data_format, output_path, "compressed_data")
    chunk_shape, shard_shape = default_chunk_config(data_format, 8)

    # create uncompressed dataset
    mag_view = (
        Dataset(ds_path, voxel_size=(1, 1, 1))
        .add_layer("color", COLOR_CATEGORY, data_format=data_format)
        .add_mag(
            "2",
            chunk_shape=chunk_shape,
            shard_shape=shard_shape,
            compress=True,
        )
    )
    mag_view.write(
        rng.integers(0, 256, (120, 140, 160), dtype=np.uint8), allow_resize=True
    )

    # open compressed dataset
    compressed_mag = Dataset.open(ds_path).get_layer("color").get_mag("2")

    with pytest.raises(ValueError, match=".*not aligned with the shard shape.*"):
        compressed_mag.write(
            absolute_offset=(10, 20, 30),
            data=rng.integers(0, 256, (10, 10, 10), dtype=np.uint8),
        )

    with pytest.raises(ValueError, match=".*not aligned with the shard shape.*"):
        compressed_mag.write(
            relative_offset=(20, 40, 60),
            data=rng.integers(0, 256, (10, 10, 10), dtype=np.uint8),
        )

    assert compressed_mag.bounding_box == BoundingBox(
        topleft=(
            0,
            0,
            0,
        ),
        size=(120 * 2, 140 * 2, 160 * 2),
    )
    # Writing unaligned data to the edge of the bounding box of the MagView does not raise an error.
    # This write operation writes unaligned data into the bottom-right corner of the MagView.
    compressed_mag.write(
        absolute_offset=(128, 128, 128),
        data=rng.integers(0, 256, (56, 76, 96), dtype=np.uint8),
    )

    # This also works for normal Views but they only use the bounding box at the time of creation as reference.
    compressed_mag.get_view().write(
        absolute_offset=(128, 128, 128),
        data=rng.integers(0, 256, (56, 76, 96), dtype=np.uint8),
    )

    # Writing aligned data does not raise a warning. Therefore, this does not fail with these strict settings.
    compressed_mag.write(data=rng.integers(0, 256, (64, 64, 64), dtype=np.uint8))


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_writing_subset_of_chunked_compressed_data(
    data_format: DataFormat, output_path: UPath
) -> None:
    ds_path = prepare_dataset_path(data_format, output_path, "compressed_data")
    chunk_shape, shard_shape = default_chunk_config(data_format, 8)

    write_data1 = rng.integers(0, 256, (100, 200, 300), dtype=np.uint8)
    write_data2 = rng.integers(0, 256, (50, 40, 30), dtype=np.uint8)
    mag_view = (
        Dataset(ds_path, voxel_size=(1, 1, 1))
        .add_layer("color", COLOR_CATEGORY, data_format=data_format)
        .add_mag(
            "1",
            chunk_shape=chunk_shape,
            shard_shape=shard_shape,
            compress=True,
        )
    )
    mag_view.write(write_data1, allow_resize=True, allow_unaligned=True)

    # open compressed dataset
    compressed_view = (
        Dataset.open(ds_path)
        .get_layer("color")
        .get_mag("1")
        .get_view(absolute_offset=(0, 0, 0), size=(100, 200, 300))
    )

    with pytest.raises(ValueError, match=".*not aligned with the shard shape.*"):
        # Easy case:
        # The aligned data (offset=(0,0,0), size=(64, 64, 64)) IS fully within the bounding box of the view
        compressed_view.write(absolute_offset=(10, 20, 30), data=write_data2)
    compressed_view.write(
        absolute_offset=(10, 20, 30), data=write_data2, allow_unaligned=True
    )

    with pytest.raises(ValueError, match=".*not aligned with the shard shape.*"):
        # Advanced case:
        # The aligned data (offset=(0,0,0), size=(128, 128, 128)) is NOT fully within the bounding box of the view
        compressed_view.write(
            absolute_offset=(10, 20, 30),
            data=rng.integers(0, 256, (90, 80, 70), dtype=np.uint8),
        )
    compressed_view.write(
        absolute_offset=(10, 20, 30),
        data=rng.integers(0, 256, (90, 80, 70), dtype=np.uint8),
        allow_unaligned=True,
    )

    np.array_equal(
        write_data2,
        compressed_view.read(absolute_offset=(10, 20, 30), size=(50, 40, 30)),
    )  # the new data was written
    np.array_equal(
        write_data1[:10, :20, :30],
        compressed_view.read(absolute_offset=(0, 0, 0), size=(10, 20, 30)),
    )  # the old data is still there


@pytest.mark.parametrize(
    "data_format", DATA_FORMATS
)  # Don't test remote storage for performance reasons (lack of sharding in zarr)
def test_for_zipped_chunks(data_format: DataFormat) -> None:
    src_dataset_path = prepare_dataset_path(
        data_format, TESTOUTPUT_DIR, "zipped_chunking_source"
    )
    dst_dataset_path = prepare_dataset_path(
        data_format, TESTOUTPUT_DIR, "zipped_chunking_target"
    )

    ds = Dataset(src_dataset_path, voxel_size=(1, 1, 2))
    mag = ds.add_layer(
        "color",
        category=COLOR_CATEGORY,
        dtype="uint8",
        num_channels=3,
        data_format=data_format,
    ).add_mag("1")
    mag.write(
        data=rng.integers(0, 256, (3, 128, 128, 128), dtype=np.uint8),
        allow_resize=True,
    )
    source_view = mag.get_view(
        absolute_offset=(0, 0, 0), size=(128, 128, 128), read_only=True
    )

    target_mag = (
        Dataset(dst_dataset_path, voxel_size=(1, 1, 2))
        .get_or_add_layer(
            "color",
            COLOR_CATEGORY,
            dtype="uint8",
            num_channels=3,
            data_format=data_format,
        )
        .get_or_add_mag(
            "1",
            chunk_shape=Vec3Int.full(8),
            shard_shape=(32 if data_format != DataFormat.Zarr else 8),
        )
    )

    target_mag.layer.bounding_box = BoundingBox((0, 0, 0), (128, 128, 128))
    target_view = target_mag.get_view(absolute_offset=(0, 0, 0), size=(128, 128, 128))

    with get_executor("sequential") as executor:
        func = named_partial(
            copy_and_transform_job, name="foo", val=42
        )  # curry the function with further arguments
        source_view.for_zipped_chunks(
            func,
            target_view=target_view,
            source_chunk_shape=(32, 32, 32),  # multiple of (wkw_file_len,) * 3
            target_chunk_shape=(32, 32, 32),  # multiple of (wkw_file_len,) * 3
            executor=executor,
        )

    np.testing.assert_array_equal(
        source_view.read() + 50,
        target_view.read(),
    )

    assure_exported_properties(ds)


def _func_invalid_target_chunk_shape_wk(args: tuple[View, View, int]) -> None:
    (_s, _t, _i) = args


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_for_zipped_chunks_invalid_target_chunk_shape_wk(
    data_format: DataFormat, output_path: UPath
) -> None:
    ds_path = prepare_dataset_path(
        data_format, output_path, "zipped_chunking_source_invalid"
    )
    chunk_shape, shard_shape = default_chunk_config(data_format, 8)
    test_cases_wk = [
        (10, 20, 30),
        (64, 64, 100),
        (64, 50, 64),
        (200, 128, 128),
    ]

    ds = Dataset(ds_path, voxel_size=(1, 1, 1))
    layer1 = ds.get_or_add_layer("color1", COLOR_CATEGORY, data_format=data_format)
    source_mag_view = layer1.get_or_add_mag(
        1, chunk_shape=chunk_shape, shard_shape=shard_shape
    )

    layer2 = ds.get_or_add_layer("color2", COLOR_CATEGORY, data_format=data_format)
    target_mag_view = layer2.get_or_add_mag(
        1, chunk_shape=chunk_shape, shard_shape=shard_shape
    )

    source_view = source_mag_view.get_view(
        absolute_offset=(0, 0, 0), size=(300, 300, 300), read_only=True
    )
    layer2.bounding_box = BoundingBox((0, 0, 0), (300, 300, 300))
    target_view = target_mag_view.get_view()

    with get_executor("sequential") as executor:
        for test_case in test_cases_wk:
            with pytest.raises(AssertionError):
                source_view.for_zipped_chunks(
                    func_per_chunk=_func_invalid_target_chunk_shape_wk,
                    target_view=target_view,
                    source_chunk_shape=test_case,
                    target_chunk_shape=test_case,
                    executor=executor,
                )

    assure_exported_properties(ds)


@pytest.mark.parametrize("output_path", OUTPUT_PATHS)
def test_invalid_chunk_shard_shape(output_path: UPath) -> None:
    ds_path = prepare_dataset_path(
        DEFAULT_DATA_FORMAT, output_path, "invalid_chunk_shape"
    )
    ds = Dataset(ds_path, voxel_size=(1, 1, 1))
    layer = ds.add_layer("color", COLOR_CATEGORY, data_format=DEFAULT_DATA_FORMAT)

    with pytest.raises(ValueError, match=".*must be a power of two.*"):
        layer.add_mag("1", chunk_shape=(3, 4, 4))

    with pytest.raises(ValueError, match=".*must be a multiple.*"):
        layer.add_mag("1", chunk_shape=(16, 16, 16), shard_shape=(8, 16, 16))

    with pytest.raises(ValueError, match=".*must be a multiple.*"):
        layer.add_mag("1", chunk_shape=(16, 16, 16), shard_shape=(8, 8, 8))

    with pytest.raises(ValueError, match=".*must be a multiple.*"):
        # also not a power-of-two shard shape
        layer.add_mag("1", chunk_shape=(16, 16, 16), shard_shape=(53, 16, 16))


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_bounding_box_on_disk(data_format: DataFormat, output_path: UPath) -> None:
    ds_path = prepare_dataset_path(data_format, output_path)
    ds = Dataset(ds_path, voxel_size=(2, 2, 1))
    chunk_shape, shard_shape = default_chunk_config(data_format, 8)
    mag = ds.add_layer("color", category="color", data_format=data_format).add_mag(
        "2-2-1", chunk_shape=chunk_shape, shard_shape=shard_shape
    )  # cube_size = 8*8 = 64

    write_positions = [
        Vec3Int(0, 0, 0),
        Vec3Int(20, 80, 120),
        Vec3Int(1000, 2000, 4000),
    ]
    data_size = Vec3Int(10, 20, 30)
    write_data = rng.integers(0, 256, tuple(data_size), dtype=np.uint8)
    for offset in write_positions:
        mag.write(
            absolute_offset=offset * mag.mag.to_vec3_int(),
            data=write_data,
            allow_resize=True,
            allow_unaligned=True,
        )

    if is_remote_path(output_path):
        with pytest.warns(UserWarning, match=".*can be slow.*"):
            bounding_boxes_on_disk = list(mag.get_bounding_boxes_on_disk())

        assert (
            len(bounding_boxes_on_disk)
            == mag.bounding_box.size.ceildiv(mag._array.info.shard_shape)
            .ceildiv(mag.mag)
            .prod()
        )
    else:
        bounding_boxes_on_disk = list(mag.get_bounding_boxes_on_disk())
        file_size = mag._get_file_dimensions()

        expected_results = set()
        for offset in write_positions:
            range_from = offset // file_size * file_size
            range_to = offset + data_size
            # enumerate all bounding boxes of the current write operation
            x_range = range(
                range_from[0],
                range_to[0],
                file_size[0],
            )
            y_range = range(
                range_from[1],
                range_to[1],
                file_size[1],
            )
            z_range = range(
                range_from[2],
                range_to[2],
                file_size[2],
            )

            for bb_offset in itertools.product(x_range, y_range, z_range):
                expected_results.add(
                    BoundingBox(bb_offset, file_size).from_mag_to_mag1(mag.mag)
                )

        assert set(bounding_boxes_on_disk) == expected_results


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_compression(data_format: DataFormat, output_path: UPath) -> None:
    new_dataset_path = prepare_dataset_path(data_format, output_path)
    ds = Dataset(new_dataset_path, voxel_size=(2, 2, 1))
    mag1 = ds.add_layer(
        "color", COLOR_CATEGORY, num_channels=3, data_format=data_format
    ).add_mag(1, compress=False)

    # writing unaligned data to an uncompressed dataset
    write_data = rng.integers(0, 256, (3, 10, 20, 30), dtype=np.uint8)
    mag1.write(write_data, absolute_offset=(60, 80, 100), allow_resize=True)

    assert not mag1._is_compressed()

    if output_path == REMOTE_TESTOUTPUT_DIR:
        # Remote datasets require a `target_path` for compression
        with pytest.raises(AssertionError):
            mag1.compress()

        compressed_dataset_path = (
            REMOTE_TESTOUTPUT_DIR / f"simple_{data_format}_dataset_compressed"
        )
        with pytest.warns(UserWarning, match=".*can be slow.*"):
            mag1.compress(
                target_path=compressed_dataset_path,
            )
        mag1 = Dataset.open(compressed_dataset_path).get_layer("color").get_mag(1)
    else:
        with get_executor("sequential") as executor:
            mag1.compress(executor=executor)

    assert mag1._is_compressed()
    assert mag1.info.data_format == data_format

    np.testing.assert_array_equal(
        write_data, mag1.read(absolute_offset=(60, 80, 100), size=(10, 20, 30))
    )

    # writing unaligned data to a compressed dataset works because the data gets padded, but it prints a warning
    mag1.write(rng.integers(0, 256, (3, 10, 20, 30), dtype=np.uint8), allow_resize=True)

    assure_exported_properties(mag1.layer.dataset)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_rechunking(data_format: DataFormat, output_path: UPath) -> None:
    new_dataset_path = prepare_dataset_path(data_format, output_path)
    ds = Dataset(new_dataset_path, voxel_size=(2, 2, 1))
    mag1 = ds.add_layer(
        "color", COLOR_CATEGORY, num_channels=3, data_format=data_format
    ).add_mag(
        1,
        compress=False,
        chunk_shape=(16, 16, 16),
        shard_shape=(16, 16, 16) if data_format == DataFormat.Zarr else (64, 64, 64),
    )

    # writing unaligned data to an uncompressed dataset
    write_data = rng.integers(0, 256, (3, 10, 20, 30), dtype=np.uint8)
    mag1.write(write_data, absolute_offset=(60, 80, 100), allow_resize=True)

    assert not mag1._is_compressed()

    if output_path == REMOTE_TESTOUTPUT_DIR:
        # Remote datasets require a `target_path` for rechunking
        with pytest.raises(AssertionError):
            mag1.rechunk()

        compressed_dataset_path = (
            REMOTE_TESTOUTPUT_DIR / f"simple_{data_format}_dataset_compressed"
        )
        with pytest.warns(UserWarning, match=".*can be slow.*"):
            mag1.rechunk(
                target_path=compressed_dataset_path,
            )
        mag1 = Dataset.open(compressed_dataset_path).get_layer("color").get_mag(1)
    else:
        with get_executor("sequential") as executor:
            mag1.rechunk(executor=executor)

    assert mag1.info.data_format == data_format
    assert mag1._is_compressed()
    assert mag1.info.chunk_shape == DEFAULT_CHUNK_SHAPE
    if data_format == DataFormat.Zarr:
        assert mag1.info.shard_shape == DEFAULT_CHUNK_SHAPE
    else:
        assert mag1.info.shard_shape == DEFAULT_SHARD_SHAPE

    np.testing.assert_array_equal(
        write_data, mag1.read(absolute_offset=(60, 80, 100), size=(10, 20, 30))
    )

    # writing unaligned data to a compressed dataset works because the data gets padded, but it prints a warning
    mag1.write(rng.integers(0, 256, (3, 10, 20, 30), dtype=np.uint8), allow_resize=True)

    assure_exported_properties(mag1.layer.dataset)


@pytest.mark.parametrize("output_path", OUTPUT_PATHS)
def test_zarr3_config(output_path: UPath) -> None:
    new_dataset_path = prepare_dataset_path(DataFormat.Zarr3, output_path)
    ds = Dataset(new_dataset_path, voxel_size=(2, 2, 1))
    mag1 = ds.add_layer(
        "color", COLOR_CATEGORY, num_channels=3, data_format=DataFormat.Zarr3
    ).add_mag(
        1,
        compress=Zarr3Config(
            codecs=(
                {"name": "bytes"},
                {"name": "gzip", "configuration": {"level": 3}},
            ),
            chunk_key_encoding={
                "name": "default",
                "configuration": {"separator": "."},
            },
        ),
    )

    # writing unaligned data to an uncompressed dataset
    write_data = rng.integers(0, 256, (3, 10, 20, 30), dtype=np.uint8)
    mag1.write(write_data, absolute_offset=(60, 80, 100), allow_resize=True)

    assert isinstance(mag1.info, Zarr3ArrayInfo)
    assert mag1.info.codecs == (
        {"name": "bytes"},
        {"name": "gzip", "configuration": {"level": 3}},
    )
    assert mag1.info.chunk_key_encoding == {
        "name": "default",
        "configuration": {"separator": "."},
    }
    assert (mag1.path / "c.0.0.0.0").exists()
    assert json.loads((mag1.path / "zarr.json").read_bytes())["codecs"][0][
        "configuration"
    ]["codecs"] == [
        {"name": "bytes"},
        {"name": "gzip", "configuration": {"level": 3}},
    ]

    np.testing.assert_array_equal(
        write_data, mag1.read(absolute_offset=(60, 80, 100), size=(10, 20, 30))
    )

    assure_exported_properties(mag1.layer.dataset)


@pytest.mark.parametrize("output_path", OUTPUT_PATHS)
def test_zarr3_sharding(output_path: UPath) -> None:
    new_dataset_path = prepare_dataset_path(DataFormat.Zarr3, output_path)
    ds = Dataset(new_dataset_path, voxel_size=(2, 2, 1))
    mag1 = ds.add_layer(
        "color", COLOR_CATEGORY, num_channels=3, data_format=DataFormat.Zarr3
    ).add_mag(1, chunk_shape=(32, 32, 32), shard_shape=(64, 64, 64))

    # writing unaligned data to an uncompressed dataset
    write_data = rng.integers(0, 256, (3, 10, 20, 30), dtype=np.uint8)
    mag1.write(write_data, absolute_offset=(60, 80, 100), allow_resize=True)

    assert (
        json.loads((mag1.path / "zarr.json").read_bytes())["codecs"][0]["name"]
        == "sharding_indexed"
    )

    np.testing.assert_array_equal(
        write_data, mag1.read(absolute_offset=(60, 80, 100), size=(10, 20, 30))
    )

    assure_exported_properties(mag1.layer.dataset)


@pytest.mark.parametrize("output_path", OUTPUT_PATHS)
def test_zarr3_no_sharding(output_path: UPath) -> None:
    new_dataset_path = prepare_dataset_path(DataFormat.Zarr3, output_path)
    ds = Dataset(new_dataset_path, voxel_size=(2, 2, 1))
    mag1 = ds.add_layer(
        "color", COLOR_CATEGORY, num_channels=3, data_format=DataFormat.Zarr3
    ).add_mag(1, chunk_shape=(32, 32, 32), shard_shape=(32, 32, 32))

    # writing unaligned data to an uncompressed dataset
    write_data = rng.integers(0, 256, (3, 10, 20, 30), dtype=np.uint8)
    mag1.write(write_data, absolute_offset=(60, 80, 100), allow_resize=True)

    # Don't set up a sharding codec, if no sharding is necessary, i.e. chunk_shape == shard_shape
    assert (
        json.loads((mag1.path / "zarr.json").read_bytes())["codecs"][0]["name"]
        != "sharding_indexed"
    )

    np.testing.assert_array_equal(
        write_data, mag1.read(absolute_offset=(60, 80, 100), size=(10, 20, 30))
    )

    assure_exported_properties(mag1.layer.dataset)


def test_can_compress_mag8() -> None:
    ds_path = prepare_dataset_path(DataFormat.WKW, TESTOUTPUT_DIR)
    ds = Dataset(ds_path, voxel_size=(1, 1, 1))

    layer = ds.add_layer("color", COLOR_CATEGORY)
    layer.bounding_box = BoundingBox((0, 0, 0), (12240, 12240, 685))
    for mag in ["1", "2-2-1", "4-4-1", "8-8-2"]:
        layer.add_mag(mag, compress=False)

    assert layer.bounding_box == BoundingBox((0, 0, 0), (12240, 12240, 685))

    mag_view = layer.get_mag("8-8-2")
    data_to_write = rng.integers(0, 256, (1, 10, 10, 10), dtype=np.uint8)
    mag_view.write(
        data_to_write, absolute_offset=(11264, 11264, 0), allow_unaligned=True
    )
    mag_view.compress()
