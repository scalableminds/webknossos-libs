"""Reader for Neuroglancer precomputed volumes (`info` + per-scale chunked
data). Inherently multiscale — an `info` file always lists at least one scale.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from upath import UPath

from ...geometry.constants import C_AXIS, X_AXIS, Y_AXIS, Z_AXIS
from .._utils.tensorstore_helpers import _make_kvstore
from ..errors import CorruptImageError
from .image_source import ReadOptions, compute_channel_selection
from .image_source_registry import register_chunked_image_source
from .tensorstore_chunked_image_source import TensorStoreChunkedImageSource

_INFO_FILE_NAME = "info"

# Neuroglancer precomputed's physical axis order is always (x, y, z, channel)
# per the format's own spec — no guessing needed, unlike Zarr/N5.
_AXES = (X_AXIS, Y_AXIS, Z_AXIS, C_AXIS)


def _read_info(path: UPath) -> Any:
    try:
        return json.loads((path / _INFO_FILE_NAME).read_bytes())
    except Exception as e:
        raise CorruptImageError(
            f"Cannot read {path / _INFO_FILE_NAME}. It is likely corrupted or "
            "not valid JSON.",
            path=path,
        ) from e


def _resolution_key(scale: dict[str, Any]) -> float:
    x, y, z = scale["resolution"]
    return float(x) * float(y) * float(z)


@register_chunked_image_source
class NeuroglancerPrecomputedImageSource(TensorStoreChunkedImageSource):
    """
    ChunkedImageSource for Neuroglancer precomputed volumes. Inherently
    multiscale; the finest scale is opened by default.
    `ReadOptions.format_options["scale"]` picks another scale (0 = finest),
    the same way `czi_channel` picks a CZI acquisition channel.

    No suffix convention exists for this format — it is only ever recognized
    via `probe_directory`.
    """

    @classmethod
    def supported_file_extensions(cls) -> set[str]:
        return set()

    @classmethod
    def probe_directory(cls, path: UPath) -> bool:
        info_path = path / _INFO_FILE_NAME
        if not info_path.is_file():
            return False
        try:
            info = json.loads(info_path.read_bytes())
        except Exception:
            return False
        return isinstance(info.get("scales"), list) and len(info["scales"]) > 0

    def __init__(self, path: UPath, options: ReadOptions) -> None:
        super().__init__(path, options)
        info = _read_info(path)

        scales = info.get("scales")
        if not isinstance(scales, list) or not scales:
            raise CorruptImageError(
                f"{path / _INFO_FILE_NAME} has no scales.", path=path
            )
        try:
            self.dtype = np.dtype(info["data_type"])
            raw_num_channels = int(info["num_channels"])
            ranked_indices = sorted(
                range(len(scales)), key=lambda i: _resolution_key(scales[i])
            )
        except (KeyError, TypeError, ValueError) as e:
            raise CorruptImageError(
                f"{path / _INFO_FILE_NAME} is not a valid neuroglancer "
                "precomputed info file.",
                path=path,
            ) from e

        rank = options.format_option("scale")
        rank = 0 if rank is None else rank
        if not (0 <= rank < len(scales)):
            raise ValueError(
                f"scale {rank} does not exist in {path}. Available: "
                f"{list(range(len(scales)))}."
            )

        self._possible_layers: dict[str, list[int]] = {}

        chosen_scale = scales[ranked_indices[rank]]
        try:
            self._x, self._y, self._z = (int(v) for v in chosen_scale["size"])
            scale_key = chosen_scale["key"]
        except (KeyError, TypeError, ValueError) as e:
            raise CorruptImageError(
                f"{path / _INFO_FILE_NAME} has an invalid scale entry "
                f"{chosen_scale!r}.",
                path=path,
            ) from e

        self._axes = _AXES
        self._ts_spec = {
            "driver": "neuroglancer_precomputed",
            "kvstore": _make_kvstore(path),
            "scale_metadata": {"key": scale_key},
        }

        (
            self.num_channels,
            self._channel,
            self._first_n_channels,
            possible_channels,
        ) = compute_channel_selection(raw_num_channels, options.channel)
        if possible_channels is not None:
            self._possible_layers["channel"] = possible_channels

    def get_layer_split_options(self) -> dict[str, list[int]] | None:
        if len(self._possible_layers) == 0:
            return None
        return self._possible_layers
