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
from webknossos.geometry.constants import C_AXIS, CXYZ_AXES, X_AXIS, Y_AXIS, Z_AXIS

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
    assert multiscale.axis_names == CXYZ_AXES
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
