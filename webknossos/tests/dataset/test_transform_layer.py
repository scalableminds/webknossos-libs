import copy
import json
from collections.abc import Iterator
from typing import cast

import attr
import numpy as np
import pytest
from cluster_tools import SequentialExecutor, get_executor
from scipy.ndimage import affine_transform as scipy_affine_transform
from upath import UPath

from tests.constants import TESTDATA_DIR
from webknossos import (
    COLOR_CATEGORY,
    AffineCoordinateTransformation,
    BoundingBox,
    Dataset,
    Layer,
    Mag,
    ThinPlateSplineCoordinateTransformation,
    Vec3Float,
    Vec3Int,
)
from webknossos.dataset.dataset import PROPERTIES_FILE_NAME
from webknossos.dataset.layer._transform_utils import (
    AbstractTransform,
    AffineTransform,
    transform,
)
from webknossos.dataset_properties import DatasetProperties
from webknossos.dataset_properties.structuring import get_dataset_converter
from webknossos.geometry.constants import X_AXIS, Y_AXIS, Z_AXIS

rng = np.random.default_rng(1234)

# Small chunk/shard shapes so that the tests exercise multiple chunk jobs
# (including bbox-truncated border chunks).
CHUNK_SHAPE = Vec3Int.full(8)
SHARD_SHAPE = Vec3Int.full(16)


def _identity(coordinates: np.ndarray) -> np.ndarray:
    return coordinates


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
    data = rng.integers(0, 256, (64, 64, 64), dtype=np.uint8)
    # Offset (8, 8, 8) is not shard-aligned, so the transform processes
    # bbox-truncated border chunks.
    input_layer = _make_input_layer(tmp_upath / "in", data, offset=(8, 8, 8))
    output_layer = _make_output_layer(tmp_upath / "out")

    with SequentialExecutor() as executor:
        written_bbox = transform(
            input_layer,
            output_layer,
            inverse_transform=_identity,
            output_bbox=input_layer.bounding_box,
            executor=executor,
        )

    assert written_bbox == input_layer.bounding_box
    assert output_layer.bounding_box == input_layer.bounding_box
    output_data = output_layer.get_mag(1).read(absolute_bounding_box=written_bbox)
    np.testing.assert_array_equal(output_data[0], data)


def test_transform_translation(tmp_upath: UPath) -> None:
    data = rng.integers(0, 256, (64, 64, 64), dtype=np.uint8)
    input_layer = _make_input_layer(tmp_upath / "in", data)
    output_layer = _make_output_layer(tmp_upath / "out")

    shift = (100, 50, 30)
    output_bbox = input_layer.bounding_box.offset(shift)
    with SequentialExecutor() as executor:
        written_bbox = transform(
            input_layer,
            output_layer,
            inverse_transform=_Translate(tuple(-s for s in shift)),
            output_bbox=output_bbox,
            executor=executor,
        )

    assert written_bbox == output_bbox
    assert output_layer.bounding_box.contains_bbox(output_bbox)
    output_data = output_layer.get_mag(1).read(absolute_bounding_box=output_bbox)
    np.testing.assert_array_equal(output_data[0], data)


def test_transform_affine_rotation(tmp_upath: UPath) -> None:
    data = rng.integers(0, 256, (32, 32, 16), dtype=np.uint8)
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
        written_bbox = transform(
            input_layer,
            output_layer,
            AffineTransform(rotation),
            output_bbox=output_bbox,
            executor=executor,
            translate_to_positive=True,
        )

    # The negative output bbox is translated into positive space.
    assert written_bbox == BoundingBox((0, 0, 0), (32, 32, 16))
    output_data = output_layer.get_mag(1).read(absolute_bounding_box=written_bbox)
    np.testing.assert_array_equal(output_data[0], np.rot90(data, k=1, axes=(0, 1)))


def test_transform_affine_scale(tmp_upath: UPath) -> None:
    data = rng.integers(0, 256, (16, 16, 16), dtype=np.uint8)
    input_layer = _make_input_layer(tmp_upath / "in", data)
    output_layer = _make_output_layer(tmp_upath / "out")

    scale = np.diag([3.0, 3.0, 3.0, 1.0])
    with SequentialExecutor() as executor:
        # No output_bbox: it is computed from the transformed input bbox corners.
        written_bbox = transform(
            input_layer, output_layer, AffineTransform(scale), executor=executor
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
    data = rng.integers(0, 256, (48, 48, 32), dtype=np.uint8)
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
        written_bbox = transform(
            input_layer,
            output_layer,
            AffineTransform(forward),
            executor=executor,
            translate_to_positive=True,
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
    data = rng.integers(1, 255, (2, 32, 32, 32), dtype=np.uint8)
    input_layer = _make_input_layer(tmp_upath / "in", data, num_channels=2)
    mask_data = np.zeros((32, 32, 32), dtype=np.uint8)
    mask_data[:16, :, :] = 1
    mask_layer = input_layer.dataset.add_layer("mask", COLOR_CATEGORY)
    mask_layer.add_mag(1, chunk_shape=CHUNK_SHAPE, shard_shape=SHARD_SHAPE).write(
        mask_data, allow_resize=True
    )
    output_layer = _make_output_layer(tmp_upath / "out", num_channels=2)

    with SequentialExecutor() as executor:
        written_bbox = transform(
            input_layer,
            output_layer,
            inverse_transform=_identity,
            output_bbox=input_layer.bounding_box,
            input_mask_layer=mask_layer,
            executor=executor,
        )

    output_data = output_layer.get_mag(1).read(absolute_bounding_box=written_bbox)
    expected = data.copy()
    expected[:, 16:, :, :] = 0
    np.testing.assert_array_equal(output_data, expected)


def test_transform_mag2(tmp_upath: UPath) -> None:
    data = rng.integers(0, 256, (32, 32, 32), dtype=np.uint8)
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
        written_bbox = transform(
            input_layer,
            output_layer,
            inverse_transform=_Translate(tuple(-s for s in shift)),
            output_bbox=output_bbox,
            executor=executor,
        )

    assert written_bbox == output_bbox
    assert output_layer.mags.keys() == {Mag(2)}
    output_data = output_layer.get_mag(2).read(absolute_bounding_box=output_bbox)
    np.testing.assert_array_equal(output_data[0], data)


@pytest.mark.skip_on_windows
def test_transform_multiprocessing(tmp_upath: UPath) -> None:
    data = rng.integers(0, 256, (64, 64, 64), dtype=np.uint8)
    input_layer = _make_input_layer(tmp_upath / "in", data)
    output_layer = _make_output_layer(tmp_upath / "out")

    shift = (16, 0, 32)
    output_bbox = input_layer.bounding_box.offset(shift)
    with get_executor("multiprocessing", max_workers=2) as executor:
        written_bbox = transform(
            input_layer,
            output_layer,
            inverse_transform=_Translate(tuple(-s for s in shift)),
            output_bbox=output_bbox,
            executor=executor,
        )

    output_data = output_layer.get_mag(1).read(absolute_bounding_box=written_bbox)
    np.testing.assert_array_equal(output_data[0], data)


@pytest.mark.parametrize("fill_value", [None, 0, 100])
def test_transform_fill_value(tmp_upath: UPath, fill_value: int | None) -> None:
    data = rng.integers(1, 255, (32, 32, 32), dtype=np.uint8)
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
        transform(
            input_layer,
            output_layer,
            inverse_transform=_Translate(tuple(-s for s in shift)),
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
    data = rng.integers(0, 256, (64, 64, 64), dtype=np.uint8)
    input_layer = _make_input_layer(tmp_upath / "in", data)
    output_layer = _make_output_layer(tmp_upath / "out")

    with SequentialExecutor() as executor:
        # buffer_shape that does not evenly divide the 16**3 job chunks, so the
        # tiling (incl. truncated tiles and threading) is exercised.
        written_bbox = transform(
            input_layer,
            output_layer,
            inverse_transform=_identity,
            output_bbox=input_layer.bounding_box,
            buffer_shape=(6, 5, 7),
            executor=executor,
        )

    output_data = output_layer.get_mag(1).read(absolute_bounding_box=written_bbox)
    np.testing.assert_array_equal(output_data[0], data)


def test_transform_negative_output_bbox(tmp_upath: UPath) -> None:
    data = rng.integers(0, 256, (32, 32, 32), dtype=np.uint8)
    input_layer = _make_input_layer(tmp_upath / "in", data)
    output_layer = _make_output_layer(tmp_upath / "out")

    output_bbox = BoundingBox((-32, -32, -32), (32, 32, 32))
    with SequentialExecutor() as executor:
        with pytest.raises(ValueError):
            transform(
                input_layer,
                output_layer,
                inverse_transform=_Translate((32, 32, 32)),
                output_bbox=output_bbox,
                translate_to_positive=False,
                executor=executor,
            )

        # With translate_to_positive, the bbox is shifted to the origin and
        # the inverse transform still receives the original (untranslated) coordinates.
        written_bbox = transform(
            input_layer,
            output_layer,
            inverse_transform=_Translate((32, 32, 32)),
            output_bbox=output_bbox,
            executor=executor,
            translate_to_positive=True,
        )

    assert written_bbox == BoundingBox((0, 0, 0), (32, 32, 32))
    output_data = output_layer.get_mag(1).read(absolute_bounding_box=written_bbox)
    np.testing.assert_array_equal(output_data[0], data)


class _Shift(AbstractTransform):
    """A custom  AbstractTransform adding a constant offset."""

    def __init__(self, offset: tuple[float, ...]) -> None:
        self.offset = np.asarray(offset, dtype=np.float64)

    def apply(self, points: np.ndarray) -> np.ndarray:
        return points + self.offset

    def inverse(self) -> "_Shift":
        return _Shift(tuple(-self.offset))


def test_transform_argument_validation(tmp_upath: UPath) -> None:
    data = rng.integers(0, 256, (16, 16, 16), dtype=np.uint8)
    input_layer = _make_input_layer(tmp_upath / "in", data)
    output_layer = _make_output_layer(tmp_upath / "out")

    # Neither transform nor inverse_transform provided.
    with pytest.raises(ValueError, match="Exactly one"):
        transform(input_layer, output_layer)

    # Both provided.
    with pytest.raises(ValueError, match="Exactly one"):
        transform(
            input_layer,
            output_layer,
            AffineTransform(np.eye(4)),
            inverse_transform=_identity,
        )


def test_transform_custom_abstract_transform(tmp_upath: UPath) -> None:
    data = rng.integers(0, 256, (32, 32, 32), dtype=np.uint8)
    input_layer = _make_input_layer(tmp_upath / "in", data)
    output_layer = _make_output_layer(tmp_upath / "out")

    # A user-defined AbstractTransform is inverted internally via its inverse method,
    # and the default output_bbox is derived from its transform_bbox.
    shift = (16, 8, 24)
    with SequentialExecutor() as executor:
        written_bbox = transform(
            input_layer, output_layer, _Shift(shift), executor=executor
        )

    assert written_bbox == input_layer.bounding_box.offset(shift)
    output_data = output_layer.get_mag(1).read(absolute_bounding_box=written_bbox)
    np.testing.assert_array_equal(output_data[0], data)


# The coordinate transformations below are unrelated to the `transform` function
# above: they are layer metadata that only affects how WEBKNOSSOS renders a layer,
# and never touch the voxel data.


def test_layer_coordinate_transformations(tmp_upath: UPath) -> None:
    ds_path = tmp_upath / "coordinate_transformations"
    ds1 = Dataset(ds_path, voxel_size=(2, 2, 1))
    layer1 = ds1.add_layer("color", COLOR_CATEGORY)
    assert layer1.coordinate_transformations == ()
    assert (
        "coordinateTransformations"
        not in json.loads((ds1.path / PROPERTIES_FILE_NAME).read_text())["dataLayers"][
            0
        ]
    )

    affine = AffineCoordinateTransformation.from_translation((10, 20, 30))
    thin_plate_spline = ThinPlateSplineCoordinateTransformation(
        source=[[0, 0, 0], [1, 2, 3], [4, 5, 6]],
        target=[[1, 1, 1], [2, 4, 6], [8, 10, 12]],
    )
    layer1.coordinate_transformations = [affine, thin_plate_spline]
    assert layer1.coordinate_transformations == (affine, thin_plate_spline)

    # Test the exact representation on disk, it must match what WEBKNOSSOS expects
    properties = json.loads((ds1.path / PROPERTIES_FILE_NAME).read_text())
    assert properties["dataLayers"][0]["coordinateTransformations"] == [
        {
            "type": "affine",
            "matrix": [
                [1.0, 0.0, 0.0, 10.0],
                [0.0, 1.0, 0.0, 20.0],
                [0.0, 0.0, 1.0, 30.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        },
        {
            "type": "thin_plate_spline",
            "correspondences": {
                "source": [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                "target": [[1.0, 1.0, 1.0], [2.0, 4.0, 6.0], [8.0, 10.0, 12.0]],
            },
        },
    ]

    # Test if the data is persisted to disk
    ds2 = Dataset.open(ds_path)
    layer2 = ds2.get_layer("color")
    assert layer2.coordinate_transformations == (affine, thin_plate_spline)
    np.testing.assert_array_equal(
        cast(
            AffineCoordinateTransformation, layer2.coordinate_transformations[0]
        ).matrix,
        affine.matrix,
    )

    # The getter returns a tuple, so the layer cannot be changed by adding to or
    # removing from what it hands out
    assert isinstance(layer2.coordinate_transformations, tuple)

    # The transformations themselves are immutable, so the layer cannot be changed
    # through a reference to one of them either
    with pytest.raises(ValueError, match="read-only"):
        cast(
            AffineCoordinateTransformation, layer2.coordinate_transformations[0]
        ).matrix[0][3] = 99
    assert layer2.coordinate_transformations == (affine, thin_plate_spline)

    # Unsetting removes the key from the properties again
    layer2.coordinate_transformations = []
    assert layer2.coordinate_transformations == ()
    assert (
        "coordinateTransformations"
        not in json.loads((ds2.path / PROPERTIES_FILE_NAME).read_text())["dataLayers"][
            0
        ]
    )

    # The properties must survive a round-trip through the file on disk
    assert ds2._properties == Dataset.open(ds2.path)._properties


def _apply_affine(
    transformation: AffineCoordinateTransformation, point: tuple[float, float, float]
) -> np.ndarray:
    matrix = transformation.matrix
    return np.round(matrix[:3, :3] @ np.array(point, dtype=float) + matrix[:3, 3], 6)


def test_affine_coordinate_transformation_builders() -> None:
    # The rotations must match the convention that WEBKNOSSOS uses, i.e. a rotation
    # around z has its sine at matrix[1][0].
    assert AffineCoordinateTransformation.from_rotation(Z_AXIS, 90).matrix[1][0] == 1.0
    np.testing.assert_array_equal(
        _apply_affine(
            AffineCoordinateTransformation.from_rotation(Z_AXIS, 90), (1, 0, 0)
        ),
        [0, 1, 0],
    )
    np.testing.assert_array_equal(
        _apply_affine(
            AffineCoordinateTransformation.from_rotation(X_AXIS, 90), (0, 1, 0)
        ),
        [0, 0, 1],
    )
    np.testing.assert_array_equal(
        _apply_affine(
            AffineCoordinateTransformation.from_rotation(Y_AXIS, 90), (0, 0, 1)
        ),
        [1, 0, 0],
    )

    # `chain` applies the receiver first and its argument afterwards
    translation = AffineCoordinateTransformation.from_translation((10, 0, 0))
    scaling = AffineCoordinateTransformation.from_scale((2, 2, 2))
    np.testing.assert_array_equal(
        _apply_affine(translation.chain(scaling), (1, 0, 0)), [22, 0, 0]
    )
    np.testing.assert_array_equal(
        _apply_affine(scaling.chain(translation), (1, 0, 0)), [12, 0, 0]
    )

    # The fluent methods are equivalent to chaining the respective transformation
    identity = AffineCoordinateTransformation.identity()
    assert identity.translate((10, 0, 0)) == identity.chain(translation)
    assert identity.scale((2, 2, 2)) == identity.chain(scaling)
    assert identity.rotate(Z_AXIS, 90) == identity.chain(
        AffineCoordinateTransformation.from_rotation(Z_AXIS, 90)
    )
    np.testing.assert_array_equal(
        _apply_affine(identity.rotate(Z_AXIS, 90).translate((5, 0, 0)), (1, 0, 0)),
        [5, 1, 0],
    )
    np.testing.assert_array_equal(
        _apply_affine(identity.flip(Y_AXIS), (3, 4, 5)), [3, -4, 5]
    )

    # The builders return new objects and never modify the receiver
    assert identity == AffineCoordinateTransformation.identity()

    with pytest.raises(ValueError, match="must be `x`, `y` or `z`"):
        AffineCoordinateTransformation.from_rotation("w", 90)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be `x`, `y` or `z`"):
        identity.flip("w")  # type: ignore[arg-type]


def test_thin_plate_spline_coordinate_transformation_pairs() -> None:
    transformation = ThinPlateSplineCoordinateTransformation.from_pairs(
        [
            [Vec3Int(0, 0, 0), Vec3Int(1, 1, 1)],
            [(1, 2, 3), (4, 5, 6)],
        ]
    )
    assert transformation.pairs == (
        (Vec3Float(0.0, 0.0, 0.0), Vec3Float(1.0, 1.0, 1.0)),
        (Vec3Float(1.0, 2.0, 3.0), Vec3Float(4.0, 5.0, 6.0)),
    )
    np.testing.assert_array_equal(transformation.source, [[0, 0, 0], [1, 2, 3]])
    np.testing.assert_array_equal(transformation.target, [[1, 1, 1], [4, 5, 6]])

    # Round-trips through `pairs`
    assert (
        ThinPlateSplineCoordinateTransformation.from_pairs(transformation.pairs)
        == transformation
    )

    # Sub-voxel precision survives, since the format stores the landmarks as floats
    sub_voxel = ThinPlateSplineCoordinateTransformation.from_pairs(
        [[(0.5, 1.25, 2.0), np.array([3.5, 4.0, 5.75])]]
    )
    assert sub_voxel.pairs == ((Vec3Float(0.5, 1.25, 2.0), Vec3Float(3.5, 4.0, 5.75)),)
    assert ThinPlateSplineCoordinateTransformation.from_pairs(sub_voxel.pairs) == (
        sub_voxel
    )

    with pytest.raises(ValueError, match="pair of a source and a target"):
        ThinPlateSplineCoordinateTransformation.from_pairs([[(0, 0, 0)]])
    with pytest.raises(ValueError, match="three floats"):
        ThinPlateSplineCoordinateTransformation.from_pairs([[(0, 0), (1, 1, 1)]])


def test_coordinate_transformations_are_immutable() -> None:
    """The transformations are values, which is what makes sharing them safe."""
    matrix = np.eye(4)
    affine = AffineCoordinateTransformation(matrix=matrix)
    thin_plate_spline = ThinPlateSplineCoordinateTransformation(
        source=[[0, 0, 0]], target=[[1, 1, 1]]
    )

    # Constructing does not take ownership of the caller's array
    assert matrix.flags.writeable
    matrix[0][3] = 99
    assert affine == AffineCoordinateTransformation.identity()

    # The stored arrays cannot be written to
    for array in [affine.matrix, thin_plate_spline.source, thin_plate_spline.target]:
        assert not array.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            array[0][0] = 99

    # ... nor can they be replaced
    with pytest.raises(attr.exceptions.FrozenAttributeError):
        affine.matrix = np.eye(4)
    with pytest.raises(attr.exceptions.FrozenAttributeError):
        thin_plate_spline.source = [[1, 1, 1]]  # type: ignore[assignment]

    # Being immutable, copying one may and does hand back the same object
    assert copy.deepcopy(affine) is affine
    assert copy.copy(affine) is affine

    # The builders never modify the transformation they are called on
    assert affine.translate((1, 2, 3)) is not affine
    assert affine == AffineCoordinateTransformation.identity()


def _leaf_exceptions(exception: BaseException) -> Iterator[BaseException]:
    """Yields the non-group exceptions of a possibly nested `ExceptionGroup`.

    `BaseExceptionGroup` is only a builtin from Python 3.11 on, therefore the groups are
    recognized by their `exceptions` attribute.
    """
    nested = getattr(exception, "exceptions", None)
    if nested is None:
        yield exception
    else:
        for child in nested:
            yield from _leaf_exceptions(child)


def test_coordinate_transformation_validation() -> None:
    with pytest.raises(ValueError, match=r"shape \(4, 4\)"):
        AffineCoordinateTransformation(matrix=np.eye(3))

    with pytest.raises(ValueError, match=r"shape \(N, 3\)"):
        ThinPlateSplineCoordinateTransformation(
            source=[[0, 0], [1, 2]], target=[[0, 0], [1, 2]]
        )

    with pytest.raises(ValueError, match="same length"):
        ThinPlateSplineCoordinateTransformation(
            source=[[0, 0, 0]], target=[[0, 0, 0], [1, 2, 3]]
        )

    with pytest.raises(ValueError, match="at least one correspondence"):
        ThinPlateSplineCoordinateTransformation(source=[], target=[])
    with pytest.raises(ValueError, match="at least one correspondence"):
        ThinPlateSplineCoordinateTransformation.from_pairs([])

    # An unsupported transformation type is rejected while reading a dataset
    data = json.loads(
        (TESTDATA_DIR / "complex_property_ds" / PROPERTIES_FILE_NAME).read_text()
    )
    data["dataLayers"][0]["coordinateTransformations"] = [
        {"type": "translation", "translation": [1, 2, 3]}
    ]
    with pytest.raises(Exception) as exc_info:
        get_dataset_converter().structure(data, DatasetProperties)
    # cattrs wraps the error in a (possibly nested) ExceptionGroup
    leaves = list(_leaf_exceptions(exc_info.value))
    assert len(leaves) == 1
    assert isinstance(leaves[0], ValueError)
    assert "`affine` or `thin_plate_spline`" in str(leaves[0])
