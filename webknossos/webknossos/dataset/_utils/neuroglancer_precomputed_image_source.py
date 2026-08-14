"""ChunkedImageSource for Neuroglancer precomputed volumes.

The format has no suffix convention at all — a precomputed volume is any
directory with an `info` JSON file at its root describing one or more
resolution `scales`, each backed by its own chunked data one level below.
This reader is therefore only ever reached via `probe_directory`, never by
suffix. tensorstore's `neuroglancer_precomputed` driver always presents the
opened array with axis order (x, y, z, channel), per the format's own spec,
so no axis guessing is needed.
"""

from __future__ import annotations

import json
import math

from upath import UPath

from ..errors import CorruptImageError
from .image_source import ReadOptions
from .image_source_registry import register_chunked_reader
from .tensorstore_chunked_source import _TensorStoreChunkedImageSource
from .tensorstore_helpers import _make_kvstore


@register_chunked_reader
class NeuroglancerPrecomputedImageSource(_TensorStoreChunkedImageSource):
    """
    ChunkedImageSource for Neuroglancer precomputed volumes, resolved to
    their finest-resolution scale by default. Pass
    `format_options={"scale": <index>}` (0 = finest, indexing the scales
    sorted by ascending voxel size) to pick another.
    """

    @classmethod
    def class_exts(cls) -> set[str]:
        return set()

    @classmethod
    def probe_directory(cls, path: UPath) -> bool:
        info_path = path / "info"
        if not info_path.is_file():
            return False
        try:
            info = json.loads(info_path.read_bytes())
        except Exception:
            return False
        scales = info.get("scales")
        return isinstance(scales, list) and len(scales) > 0

    def __init__(self, path: UPath, options: ReadOptions) -> None:
        super().__init__(path, options)
        self._num_scale_levels = 1

        try:
            info = json.loads((path / "info").read_bytes())
            scales = info.get("scales") or []
            if not scales:
                raise CorruptImageError(
                    f"The neuroglancer precomputed info file at {path} lists "
                    "no scales.",
                    path=path,
                )
            self._num_scale_levels = len(scales)
            # Sorted finest (smallest voxel size) first — the order
            # get_possible_layers()["scale"] promises, and what "scale" in
            # format_options indexes into. Does not assume `scales` is
            # already ordered this way.
            scale_order = sorted(
                range(len(scales)), key=lambda i: math.prod(scales[i]["resolution"])
            )
            requested_scale = options.format_option("scale")
            if requested_scale is not None:
                assert 0 <= requested_scale < len(scales), (
                    f"Selected scale {requested_scale} (0-indexed), but the "
                    f"neuroglancer precomputed volume at {path} only has "
                    f"{len(scales)} scales."
                )
                chosen_index = scale_order[requested_scale]
            else:
                chosen_index = scale_order[0]
            chosen_key = scales[chosen_index]["key"]

            self._ts_spec = {
                "driver": "neuroglancer_precomputed",
                "kvstore": _make_kvstore(path),
                "scale_metadata": {"key": chosen_key},
            }
            array = self._open_array()
            possible_channels = self._finish_init_from_array(
                array, ("x", "y", "z", "c"), options
            )
        except CorruptImageError:
            raise
        except Exception as e:
            raise CorruptImageError(
                f"Cannot open neuroglancer precomputed volume {path}. It is "
                "likely corrupted or not a valid precomputed volume.",
                path=path,
            ) from e

        self._possible_layers: dict[str, list[int]] = {}
        if possible_channels is not None:
            self._possible_layers["channel"] = possible_channels
        if self._num_scale_levels > 1:
            self._possible_layers["scale"] = list(range(self._num_scale_levels))

    def get_possible_layers(self) -> dict[str, list[int]] | None:
        return self._possible_layers or None
