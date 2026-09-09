import pytest
from upath import UPath

from webknossos.dataset import (
    CorruptImageError,
    UnsupportedImageDataError,
    UnsupportedImageFormatError,
)
from webknossos.dataset._image_conversion.ome_zarr_helpers import (
    OmeChannelMetadata,
    layer_split_label,
    resolve_ome_multiscale,
    suggested_coordinate_transformations,
    suggested_view_configuration,
    suggested_voxel_size,
)
from webknossos.dataset_properties import (
    AffineCoordinateTransformation,
    LayerViewConfiguration,
    LengthUnit,
    VoxelSize,
)
from webknossos.geometry.constants import C_AXIS, X_AXIS, Y_AXIS, Z_AXIS

_PATH = UPath("test.ome.zarr")

_CZYX_AXES = [
    {"name": C_AXIS, "type": "channel"},
    {"name": Z_AXIS, "type": "space"},
    {"name": Y_AXIS, "type": "space"},
    {"name": X_AXIS, "type": "space"},
]


def _dataset(path: str, scale: list[float]) -> dict:
    return {
        "path": path,
        "coordinateTransformations": [{"type": "scale", "scale": scale}],
    }


def _multiscale_attributes(
    datasets: list[dict],
    *,
    axes: list[dict] | None = _CZYX_AXES,
    version: str = "0.4",
    omero: dict | None = None,
) -> dict:
    attributes: dict = {
        "multiscales": [
            {
                "version": version,
                "datasets": datasets,
                **({"axes": axes} if axes is not None else {}),
            }
        ]
    }
    if omero is not None:
        attributes["omero"] = omero
    return attributes


def _dataset_06(
    path: str, scale: list[float], translation: list[float] | None = None
) -> dict:
    endpoints = {"input": {"path": path}, "output": {"name": "intrinsic"}}
    if translation is None:
        return {
            "path": path,
            "coordinateTransformations": [
                {"type": "scale", "scale": scale, **endpoints}
            ],
        }
    return {
        "path": path,
        "coordinateTransformations": [
            {
                "type": "sequence",
                **endpoints,
                "transformations": [
                    {"type": "scale", "scale": scale},
                    {"type": "translation", "translation": translation},
                ],
            }
        ],
    }


def _multiscale_06_attributes(
    datasets: list[dict],
    *,
    coordinate_systems: list[dict] | None = None,
    version: str = "0.6",
    version_in_multiscale: bool = False,
    top_level_transformations: list[dict] | None = None,
    omero: dict | None = None,
) -> dict:
    if coordinate_systems is None:
        coordinate_systems = [{"name": "intrinsic", "axes": _CZYX_AXES}]
    multiscale: dict = {
        "coordinateSystems": coordinate_systems,
        "datasets": datasets,
    }
    if top_level_transformations is not None:
        multiscale["coordinateTransformations"] = top_level_transformations
    attributes: dict = {"multiscales": [multiscale]}
    if version_in_multiscale:
        multiscale["version"] = version
    else:
        attributes["version"] = version
    if omero is not None:
        attributes["omero"] = omero
    return attributes


def test_resolve_ome_multiscale_ranks_by_spatial_resolution_regardless_of_list_order() -> (
    None
):
    attributes = _multiscale_attributes(
        [
            _dataset("1", [1.0, 1.0, 2.0, 2.0]),  # listed first, but coarser
            _dataset("0", [1.0, 1.0, 1.0, 1.0]),
        ]
    )

    multiscale = resolve_ome_multiscale(attributes, path=_PATH)

    assert multiscale.dataset_paths == ("0", "1")
    assert multiscale.axis_names == (C_AXIS, Z_AXIS, Y_AXIS, X_AXIS)
    assert multiscale.channels is None


def test_resolve_ome_multiscale_ignores_channel_axis_scale_when_ranking() -> None:
    # The channel axis' own scale factor (10.0) must not affect the ranking —
    # only the spatial (z/y/x) factors do.
    attributes = _multiscale_attributes(
        [
            _dataset("0", [10.0, 1.0, 1.0, 1.0]),
            _dataset("1", [10.0, 1.0, 2.0, 2.0]),
        ]
    )

    multiscale = resolve_ome_multiscale(attributes, path=_PATH)

    assert multiscale.dataset_paths == ("0", "1")


def test_resolve_ome_multiscale_without_axes_uses_every_scale_factor() -> None:
    # Zarr v2 permits omitting "axes"; every scale entry is then treated as
    # spatial for ranking purposes, and no axis_names can be derived.
    attributes = _multiscale_attributes(
        [_dataset("1", [2.0, 2.0]), _dataset("0", [1.0, 1.0])],
        axes=None,
    )

    multiscale = resolve_ome_multiscale(attributes, path=_PATH)

    assert multiscale.dataset_paths == ("0", "1")
    assert multiscale.axis_names is None


def test_resolve_ome_multiscale_reads_version_from_multiscale_entry() -> None:
    # NGFF 0.4 (Zarr v2) carries "version" inside the multiscale entry rather
    # than on the enclosing attributes.
    attributes = {
        "multiscales": [
            {"version": "0.4", "datasets": [_dataset("0", [1.0, 1.0, 1.0])]}
        ]
    }

    multiscale = resolve_ome_multiscale(attributes, path=_PATH)

    assert multiscale.dataset_paths == ("0",)


@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_resolve_ome_multiscale_accepts_supported_versions(version: str) -> None:
    attributes = _multiscale_attributes(
        [_dataset("0", [1.0, 1.0, 1.0, 1.0])], version=version
    )

    multiscale = resolve_ome_multiscale(attributes, path=_PATH)

    assert multiscale.dataset_paths == ("0",)


def test_resolve_ome_multiscale_rejects_unsupported_version() -> None:
    attributes = _multiscale_attributes(
        [_dataset("0", [1.0, 1.0, 1.0, 1.0])], version="0.3"
    )

    with pytest.raises(UnsupportedImageFormatError):
        resolve_ome_multiscale(attributes, path=_PATH)


def test_resolve_ome_multiscale_requires_multiscales_metadata() -> None:
    with pytest.raises(UnsupportedImageFormatError):
        resolve_ome_multiscale({}, path=_PATH)


def test_resolve_ome_multiscale_requires_at_least_one_dataset() -> None:
    attributes = _multiscale_attributes([])

    with pytest.raises(CorruptImageError):
        resolve_ome_multiscale(attributes, path=_PATH)


def test_resolve_ome_multiscale_requires_a_scale_transform() -> None:
    attributes = _multiscale_attributes(
        [{"path": "0", "coordinateTransformations": [{"type": "identity"}]}]
    )

    with pytest.raises(CorruptImageError):
        resolve_ome_multiscale(attributes, path=_PATH)


def test_resolve_ome_multiscale_rejects_unsupported_axis() -> None:
    attributes = _multiscale_attributes(
        [_dataset("0", [1.0, 1.0])],
        axes=[
            {"name": "q", "type": "space"},  # not one of t/c/z/y/x
            {"name": X_AXIS, "type": "space"},
        ],
    )

    with pytest.raises(UnsupportedImageDataError):
        resolve_ome_multiscale(attributes, path=_PATH)


def test_resolve_ome_multiscale_parses_omero_channels() -> None:
    omero = {
        "channels": [
            {"color": "0000FF", "label": "DAPI"},
            {"label": "GFP"},
        ]
    }
    attributes = _multiscale_attributes(
        [_dataset("0", [1.0, 1.0, 1.0, 1.0])], omero=omero
    )

    multiscale = resolve_ome_multiscale(attributes, path=_PATH)

    assert multiscale.channels is not None
    assert [c.label for c in multiscale.channels] == ["DAPI", "GFP"]
    assert multiscale.channels[0].view_configuration is not None
    assert multiscale.channels[0].view_configuration.color == (0, 0, 255)


def test_resolve_ome_multiscale_without_omero_has_no_channels() -> None:
    attributes = _multiscale_attributes([_dataset("0", [1.0, 1.0, 1.0, 1.0])])

    multiscale = resolve_ome_multiscale(attributes, path=_PATH)

    assert multiscale.channels is None


_view_configuration = LayerViewConfiguration(color=(255, 0, 0))
_CHANNELS = (
    OmeChannelMetadata(_view_configuration, "DAPI"),
    OmeChannelMetadata(None, None),
)


def test_suggested_view_configuration_uses_pinned_channel() -> None:
    assert (
        suggested_view_configuration(_CHANNELS, 0, num_channels=1)
        is _view_configuration
    )
    assert suggested_view_configuration(_CHANNELS, 1, num_channels=1) is None


def test_suggested_view_configuration_defaults_to_channel_0_when_unpinned_and_single() -> (
    None
):
    assert (
        suggested_view_configuration(_CHANNELS, None, num_channels=1)
        is _view_configuration
    )


def test_suggested_view_configuration_is_none_when_unpinned_and_multiple_channels() -> (
    None
):
    # Ambiguous which channel is meant, so no suggestion is made.
    assert suggested_view_configuration(_CHANNELS, None, num_channels=2) is None


def test_suggested_view_configuration_is_none_for_out_of_range_channel() -> None:
    assert suggested_view_configuration(_CHANNELS, 5, num_channels=1) is None


def test_suggested_view_configuration_is_none_without_channels_metadata() -> None:
    assert suggested_view_configuration(None, 0, num_channels=1) is None


def test_layer_split_label_only_resolves_for_channel_key() -> None:
    assert layer_split_label(_CHANNELS, "scale", 0) is None
    assert layer_split_label(_CHANNELS, "channel", 0) == "DAPI"


def test_layer_split_label_is_none_without_channels_metadata() -> None:
    assert layer_split_label(None, "channel", 0) is None


def test_layer_split_label_is_none_for_out_of_range_value() -> None:
    assert layer_split_label(_CHANNELS, "channel", 5) is None


def test_layer_split_label_is_none_when_channel_has_no_label() -> None:
    assert layer_split_label(_CHANNELS, "channel", 1) is None


def test_layer_split_label_sanitizes_disallowed_characters() -> None:
    channels = (OmeChannelMetadata(None, "Hyb probe #1!"),)
    assert layer_split_label(channels, "channel", 0) == "Hybprobe1"


def test_layer_split_label_is_none_when_label_sanitizes_to_nothing() -> None:
    # A label made up entirely of disallowed characters (and leading dots,
    # stripped separately) falls back to the caller's default naming.
    channels = (OmeChannelMetadata(None, "...!!!"),)
    assert layer_split_label(channels, "channel", 0) is None


def test_resolve_ome_multiscale_06_ranks_by_spatial_resolution() -> None:
    attributes = _multiscale_06_attributes(
        [
            _dataset_06("2", [1.0, 4.0, 4.0, 4.0], [0.0, 1.5, 1.5, 1.5]),
            _dataset_06("0", [1.0, 1.0, 1.0, 1.0]),
            _dataset_06("1", [1.0, 2.0, 2.0, 2.0], [0.0, 0.5, 0.5, 0.5]),
        ]
    )

    multiscale = resolve_ome_multiscale(attributes, path=_PATH)

    assert multiscale.dataset_paths == ("0", "1", "2")
    assert multiscale.axis_names == (C_AXIS, Z_AXIS, Y_AXIS, X_AXIS)
    assert multiscale.transforms[1].scale == (1.0, 2.0, 2.0, 2.0)
    assert multiscale.transforms[1].translation == (0.0, 0.5, 0.5, 0.5)


@pytest.mark.parametrize("version", ["0.6", "0.6rc1"])
def test_resolve_ome_multiscale_accepts_06_versions(version: str) -> None:
    attributes = _multiscale_06_attributes(
        [_dataset_06("0", [1.0, 1.0, 1.0, 1.0])], version=version
    )

    multiscale = resolve_ome_multiscale(attributes, path=_PATH)

    assert multiscale.dataset_paths == ("0",)


def test_resolve_ome_multiscale_rejects_06_version_inside_the_multiscale_entry() -> (
    None
):
    attributes = _multiscale_06_attributes(
        [_dataset_06("0", [1.0, 1.0, 1.0, 1.0])], version_in_multiscale=True
    )

    with pytest.raises(UnsupportedImageFormatError):
        resolve_ome_multiscale(attributes, path=_PATH)


def test_resolve_ome_multiscale_06_accepts_an_identity_transform() -> None:
    attributes = _multiscale_06_attributes(
        [{"path": "0", "coordinateTransformations": [{"type": "identity"}]}]
    )

    multiscale = resolve_ome_multiscale(attributes, path=_PATH)

    assert multiscale.transforms[0].scale == (1.0, 1.0, 1.0, 1.0)
    assert multiscale.transforms[0].translation == (0.0, 0.0, 0.0, 0.0)


@pytest.mark.parametrize("reverse", [False, True])
def test_resolve_ome_multiscale_06_flattens_a_sequence(reverse: bool) -> None:
    scale = {"type": "scale", "scale": [1.0, 2.0, 2.0, 2.0]}
    translation = {"type": "translation", "translation": [0.0, 3.0, 3.0, 3.0]}
    transformations = [translation, scale] if reverse else [scale, translation]
    attributes = _multiscale_06_attributes(
        [
            {
                "path": "0",
                "coordinateTransformations": [
                    {"type": "sequence", "transformations": transformations}
                ],
            }
        ]
    )

    multiscale = resolve_ome_multiscale(attributes, path=_PATH)

    assert multiscale.transforms[0].scale == (1.0, 2.0, 2.0, 2.0)
    # Scaling after the translation scales the translation too.
    expected = (0.0, 6.0, 6.0, 6.0) if reverse else (0.0, 3.0, 3.0, 3.0)
    assert multiscale.transforms[0].translation == expected


def test_resolve_ome_multiscale_06_rejects_an_unsupported_transform() -> None:
    attributes = _multiscale_06_attributes(
        [
            {
                "path": "0",
                "coordinateTransformations": [
                    {"type": "affine", "affine": [[1, 0, 0, 0]]}
                ],
            }
        ]
    )

    with pytest.raises(UnsupportedImageDataError):
        resolve_ome_multiscale(attributes, path=_PATH)


def test_resolve_ome_multiscale_06_takes_the_axes_of_the_named_coordinate_system() -> (
    None
):
    attributes = _multiscale_06_attributes(
        [_dataset_06("0", [1.0, 1.0, 1.0, 1.0])],
        coordinate_systems=[
            {"name": "physical", "axes": [{"name": "q", "type": "space"}]},
            {"name": "intrinsic", "axes": _CZYX_AXES},
        ],
    )

    multiscale = resolve_ome_multiscale(attributes, path=_PATH)

    assert multiscale.axis_names == (C_AXIS, Z_AXIS, Y_AXIS, X_AXIS)


def test_resolve_ome_multiscale_keeps_the_translation_of_a_0_4_dataset() -> None:
    attributes = _multiscale_attributes(
        [
            {
                "path": "0",
                "coordinateTransformations": [
                    {"type": "scale", "scale": [1.0, 1.0, 1.0, 1.0]},
                    {"type": "translation", "translation": [0.0, 7.0, 8.0, 9.0]},
                ],
            }
        ]
    )

    multiscale = resolve_ome_multiscale(attributes, path=_PATH)

    assert multiscale.dataset_paths == ("0",)
    assert multiscale.transforms[0].translation == (0.0, 7.0, 8.0, 9.0)


_MICROMETER_AXES = [
    {"name": C_AXIS, "type": "channel"},
    {"name": Z_AXIS, "type": "space", "unit": "micrometer"},
    {"name": Y_AXIS, "type": "space", "unit": "micrometer"},
    {"name": X_AXIS, "type": "space", "unit": "micrometer"},
]


def _micrometer_multiscale() -> dict:
    return _multiscale_06_attributes(
        [
            _dataset_06("0", [1.0, 0.1, 0.5, 0.5], [0.0, 0.2, 1.0, 1.5]),
            _dataset_06("1", [1.0, 0.2, 1.0, 1.0], [0.0, 0.4, 2.0, 3.0]),
        ],
        coordinate_systems=[{"name": "intrinsic", "axes": _MICROMETER_AXES}],
    )


def test_suggested_voxel_size_uses_the_scale_of_the_requested_rank() -> None:
    multiscale = resolve_ome_multiscale(_micrometer_multiscale(), path=_PATH)

    assert suggested_voxel_size(multiscale, 0) == VoxelSize(
        (0.5, 0.5, 0.1), LengthUnit.MICROMETER
    )
    assert suggested_voxel_size(multiscale, 1) == VoxelSize(
        (1.0, 1.0, 0.2), LengthUnit.MICROMETER
    )


def test_suggested_voxel_size_normalizes_mixed_units() -> None:
    axes = [
        {"name": C_AXIS, "type": "channel"},
        {"name": Z_AXIS, "type": "space", "unit": "micrometer"},
        {"name": Y_AXIS, "type": "space", "unit": "nanometer"},
        {"name": X_AXIS, "type": "space", "unit": "nanometer"},
    ]
    attributes = _multiscale_06_attributes(
        [_dataset_06("0", [1.0, 0.1, 20.0, 20.0])],
        coordinate_systems=[{"name": "intrinsic", "axes": axes}],
    )

    multiscale = resolve_ome_multiscale(attributes, path=_PATH)

    assert suggested_voxel_size(multiscale, 0) == VoxelSize(
        (20.0, 20.0, 100.0), LengthUnit.NANOMETER
    )


def test_suggested_voxel_size_without_units_is_none() -> None:
    multiscale = resolve_ome_multiscale(
        _multiscale_06_attributes([_dataset_06("0", [1.0, 1.0, 1.0, 1.0])]),
        path=_PATH,
    )

    assert suggested_voxel_size(multiscale, 0) is None


def test_suggested_voxel_size_with_an_unknown_unit_is_none() -> None:
    axes = [{"name": X_AXIS, "type": "space", "unit": "furlong"}]
    attributes = _multiscale_06_attributes(
        [_dataset_06("0", [1.0])],
        coordinate_systems=[{"name": "intrinsic", "axes": axes}],
    )
    multiscale = resolve_ome_multiscale(attributes, path=_PATH)

    with pytest.warns(UserWarning, match="furlong"):
        assert suggested_voxel_size(multiscale, 0) is None


def test_suggested_voxel_size_of_a_2d_group_has_a_z_factor_of_one() -> None:
    axes = [
        {"name": Y_AXIS, "type": "space", "unit": "nanometer"},
        {"name": X_AXIS, "type": "space", "unit": "nanometer"},
    ]
    attributes = _multiscale_06_attributes(
        [_dataset_06("0", [11.0, 12.0])],
        coordinate_systems=[{"name": "intrinsic", "axes": axes}],
    )

    multiscale = resolve_ome_multiscale(attributes, path=_PATH)

    assert suggested_voxel_size(multiscale, 0) == VoxelSize(
        (12.0, 11.0, 1.0), LengthUnit.NANOMETER
    )


def test_suggested_coordinate_transformations_are_the_translation_in_voxels() -> None:
    multiscale = resolve_ome_multiscale(_micrometer_multiscale(), path=_PATH)

    transformations = suggested_coordinate_transformations(multiscale, 0)

    assert transformations == (
        AffineCoordinateTransformation.from_translation((3.0, 2.0, 2.0)),
    )
    # The coarse level sits at the same physical position, so its translation
    # in its own (twice as large) voxels is the same.
    assert suggested_coordinate_transformations(multiscale, 1) == transformations


def test_suggested_coordinate_transformations_without_a_translation_are_none() -> None:
    multiscale = resolve_ome_multiscale(
        _multiscale_06_attributes([_dataset_06("0", [1.0, 1.0, 1.0, 1.0])]),
        path=_PATH,
    )

    assert suggested_coordinate_transformations(multiscale, 0) is None


def test_top_level_transformation_is_folded_in() -> None:
    attributes = _multiscale_06_attributes(
        [_dataset_06("0", [1.0, 1.0, 2.0, 2.0], [0.0, 0.0, 1.0, 1.0])],
        coordinate_systems=[{"name": "intrinsic", "axes": _MICROMETER_AXES}],
        top_level_transformations=[
            {"type": "scale", "scale": [1.0, 1.0, 10.0, 10.0]},
            {"type": "translation", "translation": [0.0, 0.0, 5.0, 5.0]},
        ],
    )
    multiscale = resolve_ome_multiscale(attributes, path=_PATH)

    assert suggested_voxel_size(multiscale, 0) == VoxelSize(
        (20.0, 20.0, 1.0), LengthUnit.MICROMETER
    )
    # (10 * 1 + 5) / (10 * 2) for x and y, nothing for z.
    assert suggested_coordinate_transformations(multiscale, 0) == (
        AffineCoordinateTransformation.from_translation((0.75, 0.75, 0.0)),
    )


def test_unsupported_top_level_transformation_is_ignored() -> None:
    attributes = _multiscale_06_attributes(
        [_dataset_06("0", [1.0, 1.0, 1.0, 1.0])],
        top_level_transformations=[{"type": "rotation", "rotation": [0.0]}],
    )

    with pytest.warns(UserWarning, match="top-level"):
        multiscale = resolve_ome_multiscale(attributes, path=_PATH)

    assert multiscale.top_level_transform is None
