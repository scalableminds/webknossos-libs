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
    suggested_view_configuration,
)
from webknossos.dataset_properties import LayerViewConfiguration
from webknossos.geometry.constants import C_AXIS, T_AXIS, X_AXIS, Y_AXIS, Z_AXIS

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


def _dataset_0_6(
    path: str,
    scale: list[float],
    translation: list[float] | None = None,
    *,
    output: str = "physical",
) -> dict:
    endpoints = {"input": {"path": path}, "output": {"name": output}}
    transform: dict
    if translation is None:
        transform = {"type": "scale", "scale": scale, **endpoints}
    else:
        transform = {
            "type": "sequence",
            **endpoints,
            "transformations": [
                {"type": "scale", "scale": scale},
                {"type": "translation", "translation": translation},
            ],
        }
    return {"path": path, "coordinateTransformations": [transform]}


def _multiscale_attributes_0_6(
    datasets: list[dict],
    *,
    coordinate_systems: list[dict] | None = None,
    omero: dict | None = None,
) -> dict:
    if coordinate_systems is None:
        coordinate_systems = [{"name": "physical", "axes": _CZYX_AXES}]
    attributes: dict = {
        "version": "0.6",
        "multiscales": [
            {"coordinateSystems": coordinate_systems, "datasets": datasets}
        ],
    }
    if omero is not None:
        attributes["omero"] = omero
    return attributes


def test_resolve_ome_multiscale_0_6_spec_example() -> None:
    # The complete multiscales example of the NGFF 0.6 spec, as `attributes.ome`.
    attributes = {
        "version": "0.6",
        "multiscales": [
            {
                "name": "example",
                "coordinateSystems": [
                    {
                        "name": "intrinsic",
                        "axes": [
                            {"name": "t", "type": "time", "unit": "millisecond"},
                            {"name": "c", "type": "channel"},
                            {"name": "z", "type": "space", "unit": "micrometer"},
                            {"name": "y", "type": "space", "unit": "micrometer"},
                            {"name": "x", "type": "space", "unit": "micrometer"},
                        ],
                    }
                ],
                "datasets": [
                    {
                        "path": "s0",
                        "coordinateTransformations": [
                            {
                                "type": "scale",
                                "scale": [0.1, 1.0, 0.5, 0.5, 0.5],
                                "input": {"path": "s0"},
                                "output": {"name": "intrinsic"},
                            }
                        ],
                    },
                    {
                        "path": "s1",
                        "coordinateTransformations": [
                            {
                                "type": "sequence",
                                "input": {"path": "s1"},
                                "output": {"name": "intrinsic"},
                                "transformations": [
                                    {
                                        "type": "scale",
                                        "scale": [0.1, 1.0, 1.0, 1.0, 1.0],
                                    },
                                    {
                                        "type": "translation",
                                        "translation": [0, 0, 0.25, 0.25, 0.25],
                                    },
                                ],
                            }
                        ],
                    },
                    {
                        "path": "s2",
                        "coordinateTransformations": [
                            {
                                "type": "sequence",
                                "input": {"path": "s2"},
                                "output": {"name": "intrinsic"},
                                "transformations": [
                                    {
                                        "type": "scale",
                                        "scale": [0.1, 1.0, 2.0, 2.0, 2.0],
                                    },
                                    {
                                        "type": "translation",
                                        "translation": [0, 0, 0.75, 0.75, 0.75],
                                    },
                                ],
                            }
                        ],
                    },
                ],
                "type": "gaussian",
            }
        ],
    }

    multiscale = resolve_ome_multiscale(attributes, path=_PATH)

    assert multiscale.dataset_paths == ("s0", "s1", "s2")
    assert multiscale.axis_names == (T_AXIS, C_AXIS, Z_AXIS, Y_AXIS, X_AXIS)


def test_resolve_ome_multiscale_0_6_ranks_sequence_scales_regardless_of_list_order() -> (
    None
):
    attributes = _multiscale_attributes_0_6(
        [
            _dataset_0_6("s2", [1.0, 4.0, 4.0, 4.0], [0, 1.5, 1.5, 1.5]),
            _dataset_0_6("s0", [1.0, 1.0, 1.0, 1.0]),
            _dataset_0_6("s1", [1.0, 2.0, 2.0, 2.0], [0, 0.5, 0.5, 0.5]),
        ]
    )

    multiscale = resolve_ome_multiscale(attributes, path=_PATH)

    assert multiscale.dataset_paths == ("s0", "s1", "s2")


def test_resolve_ome_multiscale_0_6_uses_the_coordinate_system_datasets_map_into() -> (
    None
):
    # The intrinsic coordinate system is the datasets' output, not the first entry.
    attributes = _multiscale_attributes_0_6(
        [_dataset_0_6("0", [1.0, 1.0, 1.0], output="intrinsic")],
        coordinate_systems=[
            {
                "name": "other",
                "axes": [
                    {"name": "y", "type": "space"},
                    {"name": "x", "type": "space"},
                ],
            },
            {"name": "intrinsic", "axes": _CZYX_AXES[1:]},
        ],
    )

    multiscale = resolve_ome_multiscale(attributes, path=_PATH)

    assert multiscale.axis_names == (Z_AXIS, Y_AXIS, X_AXIS)


def test_resolve_ome_multiscale_0_6_accepts_identity_transform() -> None:
    attributes = _multiscale_attributes_0_6(
        [
            {
                "path": "0",
                "coordinateTransformations": [
                    {
                        "type": "identity",
                        "input": {"path": "0"},
                        "output": {"name": "physical"},
                    }
                ],
            }
        ]
    )

    multiscale = resolve_ome_multiscale(attributes, path=_PATH)

    assert multiscale.dataset_paths == ("0",)


def test_resolve_ome_multiscale_0_6_parses_omero_channels() -> None:
    attributes = _multiscale_attributes_0_6(
        [_dataset_0_6("0", [1.0, 1.0, 1.0, 1.0])],
        omero={"channels": [{"color": "00FF00", "label": "GFP"}]},
    )

    multiscale = resolve_ome_multiscale(attributes, path=_PATH)

    assert multiscale.channels is not None
    assert multiscale.channels[0].label == "GFP"


@pytest.mark.parametrize(
    "transform",
    [
        {"type": "affine", "affine": [[1, 0, 0], [0, 1, 0]]},
        {
            "type": "sequence",
            "transformations": [
                {"type": "scale", "scale": [1.0, 1.0, 1.0, 1.0]},
                {"type": "rotation", "rotation": [[1, 0], [0, 1]]},
            ],
        },
    ],
    ids=["affine", "sequence-with-rotation"],
)
def test_resolve_ome_multiscale_0_6_rejects_unsupported_dataset_transforms(
    transform: dict,
) -> None:
    transform = {**transform, "input": {"path": "0"}, "output": {"name": "physical"}}
    attributes = _multiscale_attributes_0_6(
        [{"path": "0", "coordinateTransformations": [transform]}]
    )

    with pytest.raises(UnsupportedImageDataError, match=transform["type"]):
        resolve_ome_multiscale(attributes, path=_PATH)


def test_resolve_ome_multiscale_0_6_rejects_unknown_output_coordinate_system() -> None:
    attributes = _multiscale_attributes_0_6(
        [_dataset_0_6("0", [1.0, 1.0, 1.0, 1.0], output="missing")]
    )

    with pytest.raises(CorruptImageError, match="missing"):
        resolve_ome_multiscale(attributes, path=_PATH)


def test_resolve_ome_multiscale_0_6_rejects_inconsistent_output_coordinate_systems() -> (
    None
):
    attributes = _multiscale_attributes_0_6(
        [
            _dataset_0_6("0", [1.0, 1.0, 1.0, 1.0]),
            _dataset_0_6("1", [1.0, 2.0, 2.0, 2.0], output="other"),
        ]
    )

    with pytest.raises(CorruptImageError, match="same coordinate system"):
        resolve_ome_multiscale(attributes, path=_PATH)


def test_resolve_ome_multiscale_0_6_rejects_multiple_dataset_transforms() -> None:
    # 0.6 requires a "sequence" instead of a flat list of transforms.
    dataset = _dataset_0_6("0", [1.0, 1.0, 1.0, 1.0])
    dataset["coordinateTransformations"].append(
        {"type": "translation", "translation": [0, 0, 0, 0]}
    )
    attributes = _multiscale_attributes_0_6([dataset])

    with pytest.raises(CorruptImageError, match="exactly one"):
        resolve_ome_multiscale(attributes, path=_PATH)


def test_resolve_ome_multiscale_rejects_scene_group_with_hint() -> None:
    attributes = {"version": "0.6", "scene": {"coordinateTransformations": []}}

    with pytest.raises(UnsupportedImageFormatError, match="scene"):
        resolve_ome_multiscale(attributes, path=_PATH)


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
