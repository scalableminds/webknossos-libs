import logging
from pathlib import Path
from typing import Tuple, cast

import numpy as np
import pytest

from wkcuber.api.dataset import Dataset
from wkcuber.api.layer import Layer, LayerCategories
from wkcuber.downsampling_utils import (
    InterpolationModes,
    downsample_cube,
    downsample_cube_job,
    get_next_mag,
    calculate_default_max_mag,
    get_previous_mag,
)
import wkw
from wkcuber.mag import Mag
from wkcuber.utils import WkwDatasetInfo, open_wkw
from wkcuber.downsampling_utils import _mode, non_linear_filter_3d
import shutil

WKW_CUBE_SIZE = 1024
CUBE_EDGE_LEN = 256

TESTOUTPUT_DIR = Path("testoutput")
TESTDATA_DIR = Path("testdata")


def read_wkw(
    wkw_info: WkwDatasetInfo, offset: Tuple[int, int, int], size: Tuple[int, int, int]
) -> np.array:
    with open_wkw(wkw_info) as wkw_dataset:
        return wkw_dataset.read(offset, size)


def test_downsample_cube() -> None:
    buffer = np.zeros((CUBE_EDGE_LEN,) * 3, dtype=np.uint8)
    buffer[:, :, :] = np.arange(0, CUBE_EDGE_LEN)

    output = downsample_cube(buffer, [2, 2, 2], InterpolationModes.MODE)

    assert output.shape == (CUBE_EDGE_LEN // 2,) * 3
    assert buffer[0, 0, 0] == 0
    assert buffer[0, 0, 1] == 1
    assert np.all(output[:, :, :] == np.arange(0, CUBE_EDGE_LEN, 2))


def test_downsample_mode() -> None:

    a = np.array([[1, 3, 4, 2, 2, 7], [5, 2, 2, 1, 4, 1], [3, 3, 2, 2, 1, 1]])

    result = _mode(a)
    expected_result = np.array([1, 3, 2, 2, 1, 1])

    assert np.all(result == expected_result)


def test_downsample_median() -> None:

    a = np.array([[1, 3, 4, 2, 2, 7], [5, 2, 2, 1, 4, 1], [3, 3, 2, 2, 1, 1]])

    result = np.median(a, axis=0)
    expected_result = np.array([3, 3, 2, 2, 2, 1])

    assert np.all(result == expected_result)


def test_non_linear_filter_reshape() -> None:
    a = np.array([[[1, 3], [1, 4]], [[4, 2], [3, 1]]], dtype=np.uint8)

    a_filtered = non_linear_filter_3d(a, [2, 2, 2], _mode)
    assert a_filtered.dtype == np.uint8
    expected_result = [1]
    assert np.all(expected_result == a_filtered)

    a = np.array([[[1, 3], [1, 4]], [[4, 3], [3, 1]]], np.uint32)

    a_filtered = non_linear_filter_3d(a, [2, 2, 1], _mode)
    assert a_filtered.dtype == np.uint32
    expected_result = [1, 3]
    assert np.all(expected_result == a_filtered)


def downsample_test_helper(use_compress: bool) -> None:
    source_path = Path("testdata", "WT1_wkw")
    target_path = TESTOUTPUT_DIR / "WT1_wkw"

    try:
        shutil.rmtree(target_path)
    except:
        pass

    source_ds = Dataset(source_path)
    source_layer = source_ds.get_layer("color")
    mag1 = source_layer.get_mag("1")

    target_ds = Dataset.create(target_path, scale=(1, 1, 1))
    target_layer = target_ds.add_layer(
        "color", LayerCategories.COLOR_TYPE, dtype_per_channel="uint8"
    )
    # The bounding box has to be set here explicitly because the downsampled data is written to a different dataset.
    target_layer.set_bounding_box(
        source_layer.get_bounding_box().topleft, source_layer.get_bounding_box().size
    )
    mag2 = target_layer._initialize_mag_from_other_mag("2", mag1, use_compress)

    offset = (WKW_CUBE_SIZE, 2 * WKW_CUBE_SIZE, 0)
    target_offset = cast(Tuple[int, int, int], tuple([o // 2 for o in offset]))
    source_size = cast(Tuple[int, int, int], (CUBE_EDGE_LEN * 2,) * 3)
    target_size = cast(Tuple[int, int, int], (CUBE_EDGE_LEN,) * 3)
    source_buffer = mag1.read(
        offset=offset,
        size=source_size,
    )[0]
    assert np.any(source_buffer != 0)

    downsample_cube_job(
        (
            mag1.get_view(offset=offset, size=source_size),
            mag2.get_view(
                offset=target_offset,
                size=target_size,
            ),
            0,
        ),
        [2, 2, 2],
        InterpolationModes.MAX,
        CUBE_EDGE_LEN,
        100,
    )

    assert np.any(source_buffer != 0)

    target_buffer = mag2.read(offset=target_offset, size=target_size)[0]
    assert np.any(target_buffer != 0)

    assert np.all(
        target_buffer
        == downsample_cube(source_buffer, [2, 2, 2], InterpolationModes.MAX)
    )


def test_downsample_cube_job() -> None:
    downsample_test_helper(False)


def test_compressed_downsample_cube_job() -> None:
    downsample_test_helper(True)


def test_downsample_multi_channel() -> None:
    offset = (0, 0, 0)
    num_channels = 3
    size = (32, 32, 10)
    source_data = (
        128 * np.random.randn(num_channels, size[0], size[1], size[2])
    ).astype("uint8")
    file_len = 32

    try:
        shutil.rmtree(TESTOUTPUT_DIR / "multi-channel-test")
    except:
        pass

    ds = Dataset.create(TESTOUTPUT_DIR / "multi-channel-test", (1, 1, 1))
    l = ds.add_layer(
        "color",
        LayerCategories.COLOR_TYPE,
        dtype_per_channel="uint8",
        num_channels=num_channels,
    )
    mag1 = l.add_mag("1", file_len=file_len)

    print("writing source_data shape", source_data.shape)
    mag1.write(source_data, offset=offset)
    assert np.any(source_data != 0)

    mag2 = l._initialize_mag_from_other_mag("2", mag1, False)

    downsample_cube_job(
        (l.get_mag("1").get_view(), l.get_mag("2").get_view(), 0),
        [2, 2, 2],
        InterpolationModes.MAX,
        CUBE_EDGE_LEN,
        100,
    )

    channels = []
    for channel_index in range(num_channels):
        channels.append(
            downsample_cube(
                source_data[channel_index], [2, 2, 2], InterpolationModes.MAX
            )
        )
    joined_buffer = np.stack(channels)

    target_buffer = mag2.read(offset=offset)
    assert np.any(target_buffer != 0)
    assert np.all(target_buffer == joined_buffer)


def test_anisotropic_mag_calculation() -> None:
    mag_tests = [
        ((10.5, 10.5, 24), Mag(1), Mag((2, 2, 1))),
        ((10.5, 10.5, 21), Mag(1), Mag((2, 2, 1))),
        ((10.5, 24, 10.5), Mag(1), Mag((2, 1, 2))),
        ((24, 10.5, 10.5), Mag(1), Mag((1, 2, 2))),
        ((10.5, 10.5, 10.5), Mag(1), Mag((2, 2, 2))),
        ((10.5, 10.5, 24), Mag((2, 2, 1)), Mag((4, 4, 1))),
        ((10.5, 10.5, 21), Mag((2, 2, 1)), Mag((4, 4, 2))),
        ((10.5, 24, 10.5), Mag((2, 1, 2)), Mag((4, 1, 4))),
        ((24, 10.5, 10.5), Mag((1, 2, 2)), Mag((1, 4, 4))),
        ((10.5, 10.5, 10.5), Mag(2), Mag(4)),
        ((320, 320, 200), Mag(1), Mag((1, 1, 2))),
        ((320, 320, 200), Mag((1, 1, 2)), Mag((2, 2, 4))),
    ]

    for i in range(len(mag_tests)):
        next_mag = get_next_mag(mag_tests[i][1], mag_tests[i][0])
        assert mag_tests[i][2] == next_mag, (
            "The next anisotropic"
            f" Magnification of {mag_tests[i][1]} with "
            f"the size {mag_tests[i][0]} should be {mag_tests[i][2]} "
            f"and not {next_mag}"
        )

    for i in range(len(mag_tests)):
        previous_mag = get_previous_mag(mag_tests[i][2], mag_tests[i][0])
        assert mag_tests[i][1] == previous_mag, (
            "The previous anisotropic"
            f" Magnification of {mag_tests[i][2]} with "
            f"the size {mag_tests[i][0]} should be {mag_tests[i][1]} "
            f"and not {previous_mag}"
        )


def test_default_max_mag() -> None:
    assert calculate_default_max_mag(dataset_size=(65536, 65536, 65536)) == Mag(1024)
    assert calculate_default_max_mag(dataset_size=(4096, 4096, 4096)) == Mag(64)
    assert calculate_default_max_mag(dataset_size=(131072, 262144, 262144)) == Mag(4096)
    assert calculate_default_max_mag(dataset_size=(32768, 32768, 32768)) == Mag(512)
    assert calculate_default_max_mag(dataset_size=(16384, 65536, 65536)) == Mag(1024)
    assert calculate_default_max_mag(dataset_size=(16384, 65536, 16384)) == Mag(1024)
    assert calculate_default_max_mag(dataset_size=(256, 256, 256)) == Mag([4, 4, 4])


def test_default_parameter() -> None:
    target_path = TESTOUTPUT_DIR / "downsaple_default"

    try:
        shutil.rmtree(target_path)
    except:
        pass

    ds = Dataset.create(target_path, scale=(1, 1, 1))
    layer = ds.add_layer(
        "color", LayerCategories.COLOR_TYPE, dtype_per_channel="uint8", num_channels=3
    )
    mag = layer.add_mag("2")
    mag.write(data=(np.random.rand(3, 10, 20, 30) * 255).astype(np.uint8))
    layer.downsample()

    # The max_mag is Mag(4) in this case (see test_default_max_mag)
    assert sorted(layer.mags.keys()) == [Mag("2"), Mag("4")]


def test_default_anisotropic_scale() -> None:
    try:
        shutil.rmtree(TESTOUTPUT_DIR / "default_anisotropic_scale")
    except:
        pass

    ds = Dataset.create(
        TESTOUTPUT_DIR / "default_anisotropic_scale", scale=(85, 85, 346)
    )
    layer = ds.add_layer("color", LayerCategories.COLOR_TYPE)
    mag = layer.add_mag(1)
    mag.write(data=(np.random.rand(10, 20, 30) * 255).astype(np.uint8))

    layer.downsample(Mag(1), None, "median", True)
    assert sorted(layer.mags.keys()) == [Mag("1"), Mag("2-2-1"), Mag("4-4-1")]


def test_downsample_mag_list() -> None:
    try:
        shutil.rmtree(TESTOUTPUT_DIR / "downsample_mag_list")
    except:
        pass

    ds = Dataset.create(TESTOUTPUT_DIR / "downsample_mag_list", scale=(1, 1, 2))
    layer = ds.add_layer("color", LayerCategories.COLOR_TYPE)
    mag = layer.add_mag(1)
    mag.write(data=(np.random.rand(10, 20, 30) * 255).astype(np.uint8))

    target_mags = [Mag([4, 4, 8]), Mag(2), Mag([32, 32, 8]), Mag(32)]  # unsorted list

    layer.downsample_mag_list(from_mag=Mag(1), target_mags=target_mags)

    for m in target_mags:
        assert m in layer.mags


def test_downsample_with_invalid_mag_list() -> None:
    try:
        shutil.rmtree(TESTOUTPUT_DIR / "downsample_mag_list")
    except:
        pass

    ds = Dataset.create(TESTOUTPUT_DIR / "downsample_mag_list", scale=(1, 1, 2))
    layer = ds.add_layer("color", LayerCategories.COLOR_TYPE)
    mag = layer.add_mag(1)
    mag.write(data=(np.random.rand(10, 20, 30) * 255).astype(np.uint8))

    with pytest.raises(AssertionError):
        layer.downsample_mag_list(
            from_mag=Mag(1),
            target_mags=[Mag(1), Mag([1, 1, 2]), Mag([2, 2, 1]), Mag(2)],
        )


def test_downsample_compressed() -> None:
    try:
        shutil.rmtree(TESTOUTPUT_DIR / "downsample_compressed")
    except:
        pass

    ds = Dataset.create(TESTOUTPUT_DIR / "downsample_compressed", scale=(1, 1, 2))
    layer = ds.add_layer("color", LayerCategories.COLOR_TYPE)
    mag = layer.add_mag(1, block_len=8, file_len=8)
    mag.write(data=(np.random.rand(80, 240, 15) * 255).astype(np.uint8))

    assert not mag._is_compressed()
    mag.compress()
    assert mag._is_compressed()

    layer.downsample(
        from_mag=Mag(1),
        max_mag=Mag(
            4
        ),  # Setting max_mag to "4" covers an edge case because the z-dimension (15) has to be rounded
    )

    # Note: this test does not check if the data is correct. This is already covered by other test cases.

    assert len(layer.mags) == 3
    assert Mag("1") in layer.mags.keys()
    assert Mag("2-2-1") in layer.mags.keys()
    assert Mag("4-4-2") in layer.mags.keys()
