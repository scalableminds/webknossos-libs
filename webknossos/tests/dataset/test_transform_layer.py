import numpy as np
import pytest
from cluster_tools import SequentialExecutor, get_executor
from scipy.ndimage import affine_transform as scipy_affine_transform
from upath import UPath

from webknossos import (
    COLOR_CATEGORY,
    BoundingBox,
    Dataset,
    Layer,
    Mag,
    Vec3Int,
)
from webknossos.dataset.layer._transform_utils import AffineTransform

# Small chunk/shard shapes so that the tests exercise multiple chunk jobs
# (including bbox-truncated border chunks).
CHUNK_SHAPE = Vec3Int.full(8)
SHARD_SHAPE = Vec3Int.full(16)


def _identity(points: np.ndarray) -> np.ndarray:
    return points


class _Translate:
    """Picklable inverse transform adding a constant offset."""

    def __init__(self, offset: tuple[float, ...]) -> None:
        self.offset = np.asarray(offset, dtype=np.float64)

    def __call__(self, points: np.ndarray) -> np.ndarray:
        return points + self.offset


def _make_input_layer(
    path: UPath,
    data: np.ndarray,
    offset: tuple[int, int, int] = (0, 0, 0),
    num_channels: int = 1,
) -> Layer:
    ds = Dataset(path, voxel_size=(1, 1, 1))
    layer = ds.add_layer("color", COLOR_CATEGORY, num_channels=num_channels)
    mag = layer.add_mag(1, chunk_shape=CHUNK_SHAPE, shard_shape=SHARD_SHAPE)
    mag.write(
        absolute_offset=offset, data=data, allow_resize=True, allow_unaligned=True
    )
    return layer


def _make_output_layer(path: UPath, num_channels: int = 1) -> Layer:
    ds = Dataset(path, voxel_size=(1, 1, 1))
    return ds.add_layer("color", COLOR_CATEGORY, num_channels=num_channels)


def test_transform_identity(tmp_upath: UPath) -> None:
    data = (np.random.rand(64, 64, 64) * 255).astype(np.uint8)
    # Offset (8, 8, 8) is not shard-aligned, so the transform processes
    # bbox-truncated border chunks.
    input_layer = _make_input_layer(tmp_upath / "in", data, offset=(8, 8, 8))
    output_layer = _make_output_layer(tmp_upath / "out")

    with SequentialExecutor() as executor:
        written_bbox = input_layer.transform(
            output_layer,
            _identity,
            output_bbox=input_layer.bounding_box,
            executor=executor,
        )

    assert written_bbox == input_layer.bounding_box
    assert output_layer.bounding_box == input_layer.bounding_box
    output_data = output_layer.get_mag(1).read(absolute_bounding_box=written_bbox)
    np.testing.assert_array_equal(output_data[0], data)


def test_transform_translation(tmp_upath: UPath) -> None:
    data = (np.random.rand(64, 64, 64) * 255).astype(np.uint8)
    input_layer = _make_input_layer(tmp_upath / "in", data)
    output_layer = _make_output_layer(tmp_upath / "out")

    shift = (100, 50, 30)
    output_bbox = input_layer.bounding_box.offset(shift)
    with SequentialExecutor() as executor:
        written_bbox = input_layer.transform(
            output_layer,
            _Translate(tuple(-s for s in shift)),
            output_bbox=output_bbox,
            executor=executor,
        )

    assert written_bbox == output_bbox
    assert output_layer.bounding_box.contains_bbox(output_bbox)
    output_data = output_layer.get_mag(1).read(absolute_bounding_box=output_bbox)
    np.testing.assert_array_equal(output_data[0], data)


def test_transform_affine_rotation(tmp_upath: UPath) -> None:
    data = (np.random.rand(32, 32, 16) * 255).astype(np.uint8)
    input_layer = _make_input_layer(tmp_upath / "in", data)
    output_layer = _make_output_layer(tmp_upath / "out")

    # 90 degree counterclockwise rotation in the xy plane about the voxel
    # centers: (x, y, z) -> (-y - 1, x, z).
    rotation = np.array(
        [
            [0, -1, 0, -1],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )
    output_bbox = BoundingBox((-32, 0, 0), (32, 32, 16))
    with SequentialExecutor() as executor:
        written_bbox = input_layer.transform_affine(
            output_layer,
            rotation,
            output_bbox=output_bbox,
            executor=executor,
        )

    # The negative output bbox is translated into positive space.
    assert written_bbox == BoundingBox((0, 0, 0), (32, 32, 16))
    output_data = output_layer.get_mag(1).read(absolute_bounding_box=written_bbox)
    np.testing.assert_array_equal(output_data[0], np.rot90(data, k=1, axes=(0, 1)))


def test_transform_affine_scale(tmp_upath: UPath) -> None:
    data = (np.random.rand(16, 16, 16) * 255).astype(np.uint8)
    input_layer = _make_input_layer(tmp_upath / "in", data)
    output_layer = _make_output_layer(tmp_upath / "out")

    scale = np.diag([3.0, 3.0, 3.0, 1.0])
    with SequentialExecutor() as executor:
        # No output_bbox: it is computed from the transformed input bbox corners.
        written_bbox = input_layer.transform_affine(
            output_layer, scale, executor=executor
        )

    assert written_bbox == BoundingBox((0, 0, 0), (48, 48, 48))
    output_data = output_layer.get_mag(1).read(absolute_bounding_box=written_bbox)[0]

    # Nearest-neighbor: output voxel i samples input voxel floor(i / 3 + 0.5); positions
    # mapping past the last input sample position (15) stay zero.
    src = np.floor(np.arange(48) / 3 + 0.5).astype(np.int64)
    valid = np.arange(48) / 3 <= 15
    expected = np.zeros((48, 48, 48), dtype=np.uint8)
    expected[np.ix_(valid, valid, valid)] = data[
        np.ix_(src[valid], src[valid], src[valid])
    ]
    np.testing.assert_array_equal(output_data, expected)


def test_transform_affine_against_scipy(tmp_upath: UPath) -> None:
    data = (np.random.rand(48, 48, 32) * 255).astype(np.uint8)
    input_layer = _make_input_layer(tmp_upath / "in", data)
    output_layer = _make_output_layer(tmp_upath / "out")

    # Rotation around z combined with anisotropic scaling and a non-integer translation
    angle = np.deg2rad(30)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    forward = np.eye(4)
    forward[:3, :3] = np.diag([1.3, 0.8, 1.1]) @ rotation
    forward[:3, 3] = (5.5, -3.25, 7.0)

    with SequentialExecutor() as executor:
        written_bbox = input_layer.transform_affine(
            output_layer, forward, executor=executor
        )

    output_data = output_layer.get_mag(1).read(absolute_bounding_box=written_bbox)[0]

    # The auto-computed output bbox is shifted into positive space; undo that shift to
    # get the original (untranslated) output coordinates. The input data starts at the
    # origin, so absolute input coordinates equal array indices.
    input_bbox = input_layer.bounding_box
    assert isinstance(input_bbox, BoundingBox)
    original_bbox = AffineTransform(forward).transform_bbox(input_bbox)
    inverse = np.linalg.inv(forward)

    # scipy.ndimage.affine_transform maps output index o to input index matrix @ o + offset
    offset = inverse[:3, :3] @ original_bbox.topleft.to_np() + inverse[:3, 3]
    expected = scipy_affine_transform(
        data,
        inverse[:3, :3],
        offset=offset,
        output_shape=tuple(written_bbox.size),
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    )

    assert np.any(expected != 0)
    np.testing.assert_array_equal(output_data, expected)


def test_transform_with_mask(tmp_upath: UPath) -> None:
    data = (np.random.rand(2, 32, 32, 32) * 254 + 1).astype(np.uint8)
    input_layer = _make_input_layer(tmp_upath / "in", data, num_channels=2)
    mask_data = np.zeros((32, 32, 32), dtype=np.uint8)
    mask_data[:16, :, :] = 1
    mask_layer = input_layer._dataset.add_layer("mask", COLOR_CATEGORY)
    mask_layer.add_mag(1, chunk_shape=CHUNK_SHAPE, shard_shape=SHARD_SHAPE).write(
        mask_data, allow_resize=True
    )
    output_layer = _make_output_layer(tmp_upath / "out", num_channels=2)

    with SequentialExecutor() as executor:
        written_bbox = input_layer.transform(
            output_layer,
            _identity,
            output_bbox=input_layer.bounding_box,
            input_mask_layer=mask_layer,
            executor=executor,
        )

    output_data = output_layer.get_mag(1).read(absolute_bounding_box=written_bbox)
    expected = data.copy()
    expected[:, 16:, :, :] = 0
    np.testing.assert_array_equal(output_data, expected)


def test_transform_mag2(tmp_upath: UPath) -> None:
    data = (np.random.rand(32, 32, 32) * 255).astype(np.uint8)
    ds = Dataset(tmp_upath / "in", voxel_size=(1, 1, 1))
    input_layer = ds.add_layer("color", COLOR_CATEGORY)
    input_layer.add_mag(2, chunk_shape=CHUNK_SHAPE, shard_shape=SHARD_SHAPE).write(
        data, allow_resize=True
    )
    output_layer = _make_output_layer(tmp_upath / "out")

    shift = (4, 2, 6)  # aligned with mag 2
    output_bbox = input_layer.bounding_box.offset(shift)
    with SequentialExecutor() as executor:
        # mag=None defaults to the finest available mag, here Mag(2)
        written_bbox = input_layer.transform(
            output_layer,
            _Translate(tuple(-s for s in shift)),
            output_bbox=output_bbox,
            executor=executor,
        )

    assert written_bbox == output_bbox
    assert output_layer.mags.keys() == {Mag(2)}
    output_data = output_layer.get_mag(2).read(absolute_bounding_box=output_bbox)
    np.testing.assert_array_equal(output_data[0], data)


@pytest.mark.skip_on_windows
def test_transform_multiprocessing(tmp_upath: UPath) -> None:
    data = (np.random.rand(64, 64, 64) * 255).astype(np.uint8)
    input_layer = _make_input_layer(tmp_upath / "in", data)
    output_layer = _make_output_layer(tmp_upath / "out")

    shift = (16, 0, 32)
    output_bbox = input_layer.bounding_box.offset(shift)
    with get_executor("multiprocessing", max_workers=2) as executor:
        written_bbox = input_layer.transform(
            output_layer,
            _Translate(tuple(-s for s in shift)),
            output_bbox=output_bbox,
            executor=executor,
        )

    output_data = output_layer.get_mag(1).read(absolute_bounding_box=written_bbox)
    np.testing.assert_array_equal(output_data[0], data)


@pytest.mark.parametrize("fill_value", [None, 0, 100])
def test_transform_fill_value(tmp_upath: UPath, fill_value: int | None) -> None:
    data = (np.random.rand(32, 32, 32) * 254 + 1).astype(np.uint8)
    input_layer = _make_input_layer(tmp_upath / "in", data)

    # Pre-fill the output layer with nonzero data covering the whole output bbox.
    output_layer = _make_output_layer(tmp_upath / "out")
    output_bbox = BoundingBox((0, 0, 0), (64, 64, 64))
    output_layer.add_mag(1, chunk_shape=CHUNK_SHAPE, shard_shape=SHARD_SHAPE).write(
        np.full((64, 64, 64), 255, dtype=np.uint8), allow_resize=True
    )

    # Shift the input into the bbox center: only [16:48] has a source. Chunks in the
    # corners have no source at all; with fill_value=None they must keep the previous
    # data, otherwise they must be set to the fill_value.
    shift = (16, 16, 16)
    with SequentialExecutor() as executor:
        input_layer.transform(
            output_layer,
            _Translate(tuple(-s for s in shift)),
            output_bbox=output_bbox,
            fill_value=fill_value,
            executor=executor,
        )

    output_data = output_layer.get_mag(1).read(absolute_bounding_box=output_bbox)[0]
    background = 255 if fill_value is None else fill_value
    expected = np.full((64, 64, 64), background, dtype=np.uint8)
    expected[16:48, 16:48, 16:48] = data
    np.testing.assert_array_equal(output_data, expected)


def test_transform_small_buffer_shape(tmp_upath: UPath) -> None:
    data = (np.random.rand(64, 64, 64) * 255).astype(np.uint8)
    input_layer = _make_input_layer(tmp_upath / "in", data)
    output_layer = _make_output_layer(tmp_upath / "out")

    with SequentialExecutor() as executor:
        # buffer_shape that does not evenly divide the 16**3 job chunks, so the
        # tiling (incl. truncated tiles and threading) is exercised.
        written_bbox = input_layer.transform(
            output_layer,
            _identity,
            output_bbox=input_layer.bounding_box,
            buffer_shape=(6, 5, 7),
            executor=executor,
        )

    output_data = output_layer.get_mag(1).read(absolute_bounding_box=written_bbox)
    np.testing.assert_array_equal(output_data[0], data)


def test_transform_negative_output_bbox(tmp_upath: UPath) -> None:
    data = (np.random.rand(32, 32, 32) * 255).astype(np.uint8)
    input_layer = _make_input_layer(tmp_upath / "in", data)
    output_layer = _make_output_layer(tmp_upath / "out")

    output_bbox = BoundingBox((-32, -32, -32), (32, 32, 32))
    with SequentialExecutor() as executor:
        with pytest.raises(ValueError):
            input_layer.transform(
                output_layer,
                _Translate((32, 32, 32)),
                output_bbox=output_bbox,
                translate_to_positive=False,
                executor=executor,
            )

        # With translate_to_positive (default), the bbox is shifted to the origin and
        # the inverse transform still receives the original (untranslated) coordinates.
        written_bbox = input_layer.transform(
            output_layer,
            _Translate((32, 32, 32)),
            output_bbox=output_bbox,
            executor=executor,
        )

    assert written_bbox == BoundingBox((0, 0, 0), (32, 32, 32))
    output_data = output_layer.get_mag(1).read(absolute_bounding_box=written_bbox)
    np.testing.assert_array_equal(output_data[0], data)
