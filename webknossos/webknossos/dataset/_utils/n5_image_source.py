"""ChunkedImageSource for N5 datasets, plain or multiscale.

N5 has no standardized multiscale metadata; this reader follows the common
ecosystem convention (Neuroglancer, BigDataViewer-N5, n5-utils, paintera) of a
group containing one dataset per resolution level, named `s0`, `s1`, ... with
`s0` the finest. N5's `attributes.json` carries no axis names, so axis roles
always come from the positional OME-NGFF-style guess.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from upath import UPath

from ..errors import CorruptImageError
from .image_source import ReadOptions
from .image_source_registry import register_chunked_reader
from .tensorstore_chunked_source import (
    _TensorStoreChunkedImageSource,
    positional_axis_roles,
)
from .tensorstore_helpers import _make_kvstore

_SCALE_LEVEL_NAME = re.compile(r"s(\d+)")
_PLAIN_DATASET_KEYS = {"dimensions", "blockSize", "dataType"}


def _read_json(path: UPath) -> dict[str, Any]:
    return json.loads(path.read_bytes())


@register_chunked_reader
class N5ImageSource(_TensorStoreChunkedImageSource):
    """
    ChunkedImageSource for N5 datasets: a plain dataset, or the common
    `s0`/`s1`/... multiscale pyramid convention, resolved to its
    finest-resolution level (`s0`) by default. Pass
    `format_options={"scale": <index>}` (0 = finest, indexing the levels in
    `s0`, `s1`, ... order) to pick another.
    """

    @classmethod
    def class_exts(cls) -> set[str]:
        return {"n5"}

    @classmethod
    def probe_directory(cls, path: UPath) -> bool:
        if (path / "attributes.json").is_file():
            return True
        return any(
            _SCALE_LEVEL_NAME.fullmatch(child.name)
            and (child / "attributes.json").is_file()
            for child in path.iterdir()
            if child.is_dir()
        )

    def __init__(self, path: UPath, options: ReadOptions) -> None:
        super().__init__(path, options)
        self._num_scale_levels = 1

        try:
            resolved_path = self._resolve_dataset(path, options)
            self._ts_spec = {"driver": "n5", "kvstore": _make_kvstore(resolved_path)}
            array = self._open_array()
            axis_roles = positional_axis_roles(array.ndim)
            possible_channels = self._finish_init_from_array(array, axis_roles, options)
        except CorruptImageError:
            raise
        except Exception as e:
            raise CorruptImageError(
                f"Cannot open N5 store {path}. It is likely corrupted or not a "
                "valid N5 dataset or multiscale group.",
                path=path,
            ) from e

        self._possible_layers: dict[str, list[int]] = {}
        if possible_channels is not None:
            self._possible_layers["channel"] = possible_channels
        if self._num_scale_levels > 1:
            self._possible_layers["scale"] = list(range(self._num_scale_levels))

    def get_possible_layers(self) -> dict[str, list[int]] | None:
        return self._possible_layers or None

    def _resolve_dataset(self, path: UPath, options: ReadOptions) -> UPath:
        attrs_path = path / "attributes.json"
        if attrs_path.is_file():
            attrs = _read_json(attrs_path)
            if _PLAIN_DATASET_KEYS <= attrs.keys():
                return path

        levels = sorted(
            (
                child
                for child in path.iterdir()
                if child.is_dir() and _SCALE_LEVEL_NAME.fullmatch(child.name)
            ),
            key=lambda child: int(child.name[1:]),
        )
        if not levels:
            raise CorruptImageError(
                f"{path} is not a valid N5 dataset or multiscale group (no "
                "attributes.json with dimensions/blockSize/dataType, and no "
                "s0/s1/... levels found).",
                path=path,
            )

        # Verify the numerically-first level really is the finest, wherever
        # downsamplingFactors metadata is available to check it against.
        factor_products: dict[int, float] = {}
        for index, level in enumerate(levels):
            level_attrs_path = level / "attributes.json"
            if level_attrs_path.is_file():
                factors = _read_json(level_attrs_path).get("downsamplingFactors")
                if factors is not None:
                    factor_products[index] = math.prod(factors)
        if (
            factor_products
            and min(factor_products, key=lambda i: factor_products[i]) != 0
        ):
            raise CorruptImageError(
                f"The N5 multiscale group at {path} does not list its finest "
                "resolution first (by downsamplingFactors); this reader only "
                "supports the s0/s1/... convention.",
                path=path,
            )

        self._num_scale_levels = len(levels)
        requested_scale = options.format_option("scale")
        if requested_scale is not None:
            assert 0 <= requested_scale < len(levels), (
                f"Selected scale {requested_scale} (0-indexed), but the N5 "
                f"multiscale group at {path} only has {len(levels)} levels."
            )
            return levels[requested_scale]
        return levels[0]
