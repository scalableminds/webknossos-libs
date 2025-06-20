import warnings
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from webknossos import (
    COLOR_CATEGORY,
    SEGMENTATION_CATEGORY,
    BoundingBox,
    Dataset,
    Mag,
    Vec3Int,
)
from webknossos.dataset._upsampling_utils import upsample_cube, upsample_cube_job
from webknossos.dataset.sampling_modes import SamplingModes
from webknossos.utils import get_executor_for_args

WKW_CUBE_SIZE = 1024
BUFFER_SHAPE = Vec3Int.full(256)


@pytest.fixture(autouse=True, scope="function")
def ignore_warnings() -> Iterator:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", module="webknossos", message=r"\[WARNING\]")
        yield


def test_upsampling(tmp_path: Path) -> None:
    ds = Dataset(tmp_path, voxel_size=(1, 1, 1))
    layer = ds.add_layer("color", COLOR_CATEGORY)
    mag = layer.add_mag([4, 4, 2])
    mag.write(
        absolute_offset=(10 * 4, 20 * 4, 40 * 2),
        data=(np.random.rand(46, 45, 27) * 255).astype(np.uint8),
        allow_resize=True,
    )
    layer.upsample(
        from_mag=Mag([4, 4, 2]),
        finest_mag=Mag(1),
        compress=False,
        sampling_mode=SamplingModes.ANISOTROPIC,
    )

    assert layer.get_mag("2").read().mean() == layer.get_mag("1").read().mean()


def test_upsample_cube() -> None:
    buffer = np.zeros(BUFFER_SHAPE, dtype=np.uint8)
    buffer[:, :, :] = np.arange(0, BUFFER_SHAPE.x)

    output = upsample_cube(buffer, [2, 2, 2])

    assert np.all(output.shape == (BUFFER_SHAPE.to_np() * 2))
    assert output[0, 0, 0] == 0
    assert output[0, 0, 1] == 0
    assert output[0, 0, 2] == 1
    assert output[0, 0, 3] == 1
    assert np.all(output[:, :, :] == np.repeat(np.arange(0, BUFFER_SHAPE.x), 2))


def upsample_test_helper(tmp_path: Path, use_compress: bool) -> None:
    ds = Dataset(tmp_path, voxel_size=(10.5, 10.5, 5))
    layer = ds.add_layer("color", COLOR_CATEGORY)
    mag2 = layer.add_mag([2, 2, 2])

    offset = Vec3Int(WKW_CUBE_SIZE, 2 * WKW_CUBE_SIZE, 0)

    mag2.write(
        absolute_offset=offset,
        data=(np.random.rand(*BUFFER_SHAPE) * 255).astype(np.uint8),
        allow_resize=True,
    )
    mag1 = layer._initialize_mag_from_other_mag("1-1-2", mag2, use_compress)

    source_buffer = mag2.read(
        absolute_offset=offset,
        size=BUFFER_SHAPE,
    )[0]
    assert np.any(source_buffer != 0)

    upsample_cube_job(
        (
            mag2.get_view(absolute_offset=offset, size=BUFFER_SHAPE),
            mag1.get_view(
                absolute_offset=offset,
                size=BUFFER_SHAPE,
            ),
            0,
        ),
        [0.5, 0.5, 1.0],
        mag1.info.shard_shape,
    )

    assert np.any(source_buffer != 0)

    target_buffer = mag1.read(absolute_offset=offset, size=BUFFER_SHAPE)[0]
    assert np.any(target_buffer != 0)

    assert np.all(target_buffer == upsample_cube(source_buffer, [2, 2, 1]))


def test_upsample_cube_job(tmp_path: Path) -> None:
    upsample_test_helper(tmp_path, False)


def test_compressed_upsample_cube_job(tmp_path: Path) -> None:
    upsample_test_helper(tmp_path, True)


def test_upsample_multi_channel(tmp_path: Path) -> None:
    num_channels = 3
    size = (32, 32, 10)
    source_data = (
        128 * np.random.randn(num_channels, size[0], size[1], size[2])
    ).astype("uint8")

    ds = Dataset(tmp_path / "multi-channel-test", (1, 1, 1))
    layer = ds.add_layer(
        "color",
        COLOR_CATEGORY,
        dtype_per_channel="uint8",
        num_channels=num_channels,
    )
    mag2 = layer.add_mag("2")

    mag2.write(
        source_data,
        allow_resize=True,
    )
    assert np.any(source_data != 0)

    layer._initialize_mag_from_other_mag("1", mag2, False)

    upsample_cube_job(
        (mag2.get_view(), layer.get_mag("1").get_view(), 0),
        [0.5, 0.5, 0.5],
        mag2.info.shard_shape,
    )

    channels = [
        upsample_cube(source_data[channel_index], [2, 2, 2])
        for channel_index in range(num_channels)
    ]
    joined_buffer = np.stack(channels)

    target_buffer = layer.get_mag("1").read()
    assert np.any(target_buffer != 0)
    assert np.all(target_buffer == joined_buffer)


def test_upsampling_non_aligned(tmp_path: Path) -> None:
    ds = Dataset(tmp_path / "test", (50, 50, 50))
    layer = ds.add_layer(
        "color", SEGMENTATION_CATEGORY, dtype_per_channel="uint8", largest_segment_id=0
    )
    layer.bounding_box = BoundingBox(topleft=(0, 0, 0), size=(8409, 10267, 5271))
    layer.add_mag(32)

    layer.upsample(
        from_mag=Mag(32),
        finest_mag=Mag(8),
        sampling_mode=SamplingModes.ISOTROPIC,
        compress=True,
    )
    # The original bbox should be unchanged
    assert layer.bounding_box == BoundingBox(
        topleft=(0, 0, 0), size=(8409, 10267, 5271)
    )


def test_upsample_nd_dataset(tmp_path: Path) -> None:
    source_path = (
        Path(__file__).parent.parent.parent / "testdata" / "4D" / "4D_series_zarr3"
    )
    target_path = tmp_path / "upsample_test"

    source_ds = Dataset.open(source_path)
    target_ds = Dataset(target_path, voxel_size=(10, 10, 10))
    source_layer = source_ds.get_layer("color")
    target_layer = target_ds.add_layer(
        "color",
        COLOR_CATEGORY,
        bounding_box=source_layer.bounding_box,
        dtype_per_channel=source_layer.dtype_per_channel,
        data_format="zarr3",
    )

    source_mag = source_layer.get_mag("2")
    with get_executor_for_args(None) as executor:
        target_layer.add_mag_as_copy(source_mag, executor=executor)
        target_layer.upsample(
            from_mag=Mag(2),
            finest_mag=Mag(1),
            executor=executor,
        )

    source_data = source_layer.get_mag("1").read()
    target_data = target_layer.get_mag("1").read()

    assert source_data.shape == target_data.shape
