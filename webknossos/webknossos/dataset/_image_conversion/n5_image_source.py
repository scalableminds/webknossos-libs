"""Reader for plain N5 datasets and the common N5 multiscale pyramid
convention (`s0`, `s1`, … subdirectories, `s0` finest — used by Neuroglancer,
BigDataViewer, n5-utils, paintera and others).
"""

from __future__ import annotations

import json
import re
from typing import Any

import tensorstore as ts
from upath import UPath

from .._utils.tensorstore_helpers import TS_CONTEXT, _make_kvstore
from ..errors import CorruptImageError
from .image_source import ReadOptions, compute_channel_selection
from .image_source_registry import register_chunked_image_source
from .tensorstore_chunked_image_source import (
    TensorStoreChunkedImageSource,
    guess_axes,
)

_ATTRIBUTES_FILE_NAME = "attributes.json"
_SCALE_DIR_PATTERN = re.compile(r"^s(\d+)$")


def _read_json(path: UPath, *, path_for_errors: UPath) -> Any:
    try:
        return json.loads(path.read_bytes())
    except Exception as e:
        raise CorruptImageError(
            f"Cannot read {path}. It is likely corrupted or not valid JSON.",
            path=path_for_errors,
        ) from e


def _is_dataset_attributes(attributes: dict[str, Any]) -> bool:
    return all(key in attributes for key in ("dimensions", "blockSize", "dataType"))


@register_chunked_image_source
class N5ImageSource(TensorStoreChunkedImageSource):
    """
    ChunkedImageSource for plain N5 datasets and N5 multiscale pyramid groups.
    For a pyramid, the finest level (`s0`) is opened by default.
    `ReadOptions.format_options["scale"]` picks another level (0 = finest),
    the same way `czi_channel` picks a CZI acquisition channel.

    N5 has no axis-name metadata, so axes always come from the positional
    OME-NGFF-style convention (see `guess_axes`).
    """

    @classmethod
    def supported_file_extensions(cls) -> set[str]:
        return {"n5"}

    @classmethod
    def probe_directory(cls, path: UPath) -> bool:
        if (path / _ATTRIBUTES_FILE_NAME).is_file():
            return True
        return bool(cls._scale_level_dirs(path))

    @staticmethod
    def _scale_level_dirs(path: UPath) -> list[tuple[int, UPath]]:
        levels: list[tuple[int, UPath]] = []
        if not path.is_dir():
            return levels
        for child in path.iterdir():
            if not child.is_dir():
                continue
            match = _SCALE_DIR_PATTERN.match(child.name)
            if match and (child / _ATTRIBUTES_FILE_NAME).is_file():
                levels.append((int(match.group(1)), child))
        return levels

    def __init__(self, path: UPath, options: ReadOptions) -> None:
        super().__init__(path, options)
        self._possible_layers: dict[str, list[int]] = {}

        resolved_path = self._resolve_dataset_path(path, options)

        try:
            self._ts_spec = {"driver": "n5", "kvstore": _make_kvstore(resolved_path)}
            array = ts.open(
                self._ts_spec, open=True, context=TS_CONTEXT, recheck_cached="open"
            ).result()
        except Exception as e:
            raise CorruptImageError(
                f"Cannot open the N5 dataset at {resolved_path} (from {path}). "
                "It is likely corrupted or not a valid N5 dataset.",
                path=path,
            ) from e

        self._axes = guess_axes(len(array.shape), axis_labels=None, path=path)

        shape = array.domain.exclusive_max
        axis_to_size = dict(zip(self._axes, shape))
        self._x = axis_to_size.get("x", 1)
        self._y = axis_to_size.get("y", 1)
        self._z = axis_to_size.get("z", 1)
        raw_num_channels = axis_to_size.get("c", 1)
        self.dtype = array.dtype.numpy_dtype

        t = axis_to_size.get("t", 1)
        self._t = t
        self._include_t_axis = t > 1
        self._fixed_timepoint = None if self._include_t_axis else 0

        (
            self.num_channels,
            self._channel,
            self._first_n_channels,
            possible_channels,
        ) = compute_channel_selection(raw_num_channels, options.channel)
        if possible_channels is not None:
            self._possible_layers["channel"] = possible_channels

    def get_possible_layers(self) -> dict[str, list[int]] | None:
        if len(self._possible_layers) == 0:
            return None
        return self._possible_layers

    def _resolve_dataset_path(self, path: UPath, options: ReadOptions) -> UPath:
        attributes_path = path / _ATTRIBUTES_FILE_NAME
        if attributes_path.is_file():
            attributes = _read_json(attributes_path, path_for_errors=path)
            if _is_dataset_attributes(attributes):
                return path

        levels = sorted(self._scale_level_dirs(path))  # [(0, .../s0), (1, .../s1), ...]
        if not levels:
            raise CorruptImageError(
                f"{path} is not a valid N5 store (no {_ATTRIBUTES_FILE_NAME} "
                "describing a dataset, and no s0/s1/... pyramid levels found).",
                path=path,
            )

        self._verify_finest_is_first(path, levels)

        rank = options.format_option("scale")
        rank = 0 if rank is None else rank
        if not (0 <= rank < len(levels)):
            raise ValueError(
                f"scale {rank} does not exist in {path}. Available: "
                f"{list(range(len(levels)))}."
            )
        if len(levels) > 1:
            self._possible_layers["scale"] = list(range(len(levels)))

        return levels[rank][1]

    def _verify_finest_is_first(
        self, path: UPath, levels: list[tuple[int, UPath]]
    ) -> None:
        """Cross-checks the `s0`-is-finest naming convention against each
        level's own `downsamplingFactors`, when present, rather than trusting
        the numbering blindly."""
        factor_products: dict[int, float] = {}
        for index, level_path in levels:
            attributes = _read_json(
                level_path / _ATTRIBUTES_FILE_NAME, path_for_errors=path
            )
            factors = attributes.get("downsamplingFactors")
            if factors is not None:
                product = 1.0
                for factor in factors:
                    product *= factor
                factor_products[index] = product
        if not factor_products:
            return
        finest_index = min(factor_products, key=lambda i: factor_products[i])
        if finest_index != levels[0][0]:
            raise CorruptImageError(
                f"{path} is an N5 pyramid whose finest level (by "
                f"downsamplingFactors) is 's{finest_index}', not "
                f"'s{levels[0][0]}' as the naming convention assumes. This "
                "reader only supports the standard s0-is-finest layout.",
                path=path,
            )
