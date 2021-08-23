import tempfile
from pathlib import Path
from typing import Tuple, cast

from wkcuber.api.dataset import Dataset
from wkcuber.api.layer import LayerCategories
from wkcuber.downsampling_utils import SamplingModes
from wkcuber.upsampling_utils import upsample_cube, upsample_cube_job
from wkcuber.mag import Mag
import numpy as np


WKW_CUBE_SIZE = 1024
CUBE_EDGE_LEN = 256

TESTOUTPUT_DIR = Path("testoutput")


def test_upsampling() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        ds = Dataset.create(Path(temp_dir), scale=(1, 1, 1))
        layer = ds.add_layer("color", "COLOR")
        mag = layer.add_mag([4, 4, 2])
        mag.write(
            offset=(10, 20, 40),
            data=(np.random.rand(46, 45, 27) * 255).astype(np.uint8),
        )
        layer.upsample(
            from_mag=Mag([4, 4, 2]),
            min_mag=Mag(1),
            compress=False,
            sampling_mode=SamplingModes.AUTO,
            buffer_edge_len=64,
            args=None,
        )


def test_upsample_cube() -> None:
    buffer = np.zeros((CUBE_EDGE_LEN,) * 3, dtype=np.uint8)
    buffer[:, :, :] = np.arange(0, CUBE_EDGE_LEN)

    output = upsample_cube(buffer, [2, 2, 2])

    assert output.shape == (CUBE_EDGE_LEN * 2,) * 3
    assert output[0, 0, 0] == 0
    assert output[0, 0, 1] == 0
    assert output[0, 0, 2] == 1
    assert output[0, 0, 3] == 1
    assert np.all(output[:, :, :] == np.repeat(np.arange(0, CUBE_EDGE_LEN), 2))


def upsample_test_helper(use_compress: bool) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        ds = Dataset.create(Path(temp_dir), scale=(10.5, 10.5, 5))
        layer = ds.add_layer("color", "COLOR")
        mag2 = layer.add_mag([2, 2, 2])

        source_offset = (WKW_CUBE_SIZE, 2 * WKW_CUBE_SIZE, 0)
        target_offset = cast(
            Tuple[int, int, int], tuple([o * 2 for o in source_offset])
        )
        source_size = (CUBE_EDGE_LEN, CUBE_EDGE_LEN, CUBE_EDGE_LEN)
        target_size = (CUBE_EDGE_LEN * 2, CUBE_EDGE_LEN * 2, CUBE_EDGE_LEN)

        mag2.write(
            offset=source_offset,
            data=(np.random.rand(*source_size) * 255).astype(np.uint8),
        )
        mag1 = layer._initialize_mag_from_other_mag("1-1-2", mag2, use_compress)

        source_buffer = mag2.read(
            offset=source_offset,
            size=source_size,
        )[0]
        assert np.any(source_buffer != 0)

        upsample_cube_job(
            (
                mag2.get_view(offset=source_offset, size=source_size),
                mag1.get_view(
                    offset=target_offset,
                    size=target_size,
                ),
                0,
            ),
            [0.5, 0.5, 1.0],
            CUBE_EDGE_LEN,
            100,
        )

        assert np.any(source_buffer != 0)

        target_buffer = mag1.read(offset=target_offset, size=target_size)[0]
        assert np.any(target_buffer != 0)

        assert np.all(target_buffer == upsample_cube(source_buffer, [2, 2, 1]))


def test_upsample_cube_job() -> None:
    upsample_test_helper(False)


def test_compressed_upsample_cube_job() -> None:
    upsample_test_helper(True)


def test_upsample_multi_channel() -> None:
    num_channels = 3
    size = (32, 32, 10)
    source_data = (
        128 * np.random.randn(num_channels, size[0], size[1], size[2])
    ).astype("uint8")
    file_len = 32

    ds = Dataset.create(TESTOUTPUT_DIR / "multi-channel-test", (1, 1, 1))
    l = ds.add_layer(
        "color",
        LayerCategories.COLOR_TYPE,
        dtype_per_channel="uint8",
        num_channels=num_channels,
    )
    mag2 = l.add_mag("2", file_len=file_len)

    mag2.write(source_data)
    assert np.any(source_data != 0)

    l._initialize_mag_from_other_mag("1", mag2, False)

    upsample_cube_job(
        (mag2.get_view(), l.get_mag("1").get_view(), 0),
        [0.5, 0.5, 0.5],
        CUBE_EDGE_LEN,
        100,
    )

    channels = []
    for channel_index in range(num_channels):
        channels.append(upsample_cube(source_data[channel_index], [2, 2, 2]))
    joined_buffer = np.stack(channels)

    target_buffer = l.get_mag("1").read()
    assert np.any(target_buffer != 0)
    assert np.all(target_buffer == joined_buffer)
