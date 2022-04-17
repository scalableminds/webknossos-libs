from pathlib import Path

import numpy as np

from webknossos import COLOR_CATEGORY, Dataset, Mag, Vec3Int
from webknossos.dataset.downsampling_utils import SamplingModes
from webknossos.dataset.upsampling_utils import upsample_cube, upsample_cube_job

WKW_CUBE_SIZE = 1024
BUFFER_SHAPE = Vec3Int.full(256)


def test_upsampling(tmp_path: Path) -> None:
    ds = Dataset(tmp_path, scale=(1, 1, 1))
    layer = ds.add_layer("color", COLOR_CATEGORY)
    mag = layer.add_mag([4, 4, 2])
    mag.write(
        absolute_offset=(10 * 4, 20 * 4, 40 * 2),
        data=(np.random.rand(46, 45, 27) * 255).astype(np.uint8),
    )
    layer.upsample(
        from_mag=Mag([4, 4, 2]),
        min_mag=Mag(1),
        compress=False,
        sampling_mode=SamplingModes.ANISOTROPIC,
        buffer_edge_len=64,
        args=None,
    )


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
    ds = Dataset(tmp_path, scale=(10.5, 10.5, 5))
    layer = ds.add_layer("color", COLOR_CATEGORY)
    mag2 = layer.add_mag([2, 2, 2])

    offset = Vec3Int(WKW_CUBE_SIZE, 2 * WKW_CUBE_SIZE, 0)

    mag2.write(
        absolute_offset=offset,
        data=(np.random.rand(*BUFFER_SHAPE) * 255).astype(np.uint8),
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
        BUFFER_SHAPE,
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
    l = ds.add_layer(
        "color",
        COLOR_CATEGORY,
        dtype_per_channel="uint8",
        num_channels=num_channels,
    )
    mag2 = l.add_mag("2", chunks_per_shard=32)

    mag2.write(source_data)
    assert np.any(source_data != 0)

    l._initialize_mag_from_other_mag("1", mag2, False)

    upsample_cube_job(
        (mag2.get_view(), l.get_mag("1").get_view(), 0),
        [0.5, 0.5, 0.5],
        BUFFER_SHAPE,
    )

    channels = []
    for channel_index in range(num_channels):
        channels.append(upsample_cube(source_data[channel_index], [2, 2, 2]))
    joined_buffer = np.stack(channels)

    target_buffer = l.get_mag("1").read()
    assert np.any(target_buffer != 0)
    assert np.all(target_buffer == joined_buffer)
