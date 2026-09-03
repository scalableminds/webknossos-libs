"""Reader for zipped OME-Zarr archives (`.ozx`, NGFF RFC-9): a single ZIP file
whose root corresponds 1:1 to a Zarr v3 OME-Zarr multiscale group — the
group's own root-level `zarr.json` MUST sit at the archive root. As with
`ZarrImageSource`, only the first `multiscales` entry is considered and the
finest resolution level is opened by default; `ReadOptions.format_options`
`"scale"` picks a different level (0 = finest).

The archive is read through tensorstore's `zip` kvstore driver, which wraps
any other kvstore (here, the file the `.ozx` itself lives at) and lets the
`zarr3` driver open a sub-array inside it directly — nothing is ever
extracted to disk.
"""

from __future__ import annotations

import json
import zipfile

import tensorstore as ts
from upath import UPath

from ...dataset_properties import LayerViewConfiguration
from ...geometry.constants import C_AXIS, X_AXIS, Y_AXIS, Z_AXIS
from .._utils.tensorstore_helpers import TS_CONTEXT, _make_kvstore
from ..errors import CorruptImageError, UnsupportedImageFormatError
from .image_source import ReadOptions, compute_channel_selection
from .image_source_registry import register_chunked_image_source
from .ome_zarr_helpers import layer_split_label as _ome_layer_split_label
from .ome_zarr_helpers import resolve_ome_multiscale
from .ome_zarr_helpers import (
    suggested_view_configuration as _ome_suggested_view_configuration,
)
from .tensorstore_chunked_image_source import TensorStoreChunkedImageSource, guess_axes

_ROOT_ZARR_JSON_KEY = "zarr.json"


@register_chunked_image_source
class OzxImageSource(TensorStoreChunkedImageSource):
    """
    ChunkedImageSource for zipped OME-Zarr archives (`.ozx`, NGFF RFC-9): a
    ZIP archive whose root is a Zarr v3 OME-Zarr multiscale group, read
    without extracting it. The finest resolution level is opened by default;
    `ReadOptions.format_options["scale"]` picks another level (0 = finest),
    the same way `ZarrImageSource` does for a directory-based OME-Zarr group.
    """

    @classmethod
    def supported_file_extensions(cls) -> set[str]:
        return {"ozx"}

    def __init__(self, path: UPath, options: ReadOptions) -> None:
        super().__init__(path, options)
        self._possible_layers: dict[str, list[int]] = {}

        # `super().__init__` already rejected a remote `path`, so plain
        # `zipfile` (rather than a tensorstore kvstore) is enough to read the
        # root metadata.
        try:
            with zipfile.ZipFile(str(path)) as archive:
                root_bytes = archive.read(_ROOT_ZARR_JSON_KEY)
        except KeyError as e:
            raise UnsupportedImageFormatError(
                f"{path} has no root-level zarr.json; RFC-9 requires a "
                "zipped OME-Zarr archive's root to correspond to the "
                "OME-Zarr root.",
                path=path,
            ) from e
        except (zipfile.BadZipFile, OSError) as e:
            raise CorruptImageError(
                f"Cannot open {path} as a ZIP archive. It is likely "
                "corrupted or not a valid .ozx file.",
                path=path,
            ) from e

        try:
            root_metadata = json.loads(root_bytes)
        except Exception as e:
            raise CorruptImageError(
                f"Cannot parse the root zarr.json of {path}. It is likely "
                "corrupted or not valid JSON.",
                path=path,
            ) from e

        node_type = root_metadata.get("node_type")
        if node_type != "group":
            raise UnsupportedImageFormatError(
                f"{path} has node_type {node_type!r} at its root; RFC-9 "
                "requires a zipped OME-Zarr archive's root to be an "
                "OME-Zarr multiscale group.",
                path=path,
            )
        attributes = root_metadata.get("attributes", {})
        ome = attributes.get("ome", attributes)
        multiscale = resolve_ome_multiscale(ome, path=path)
        self._omero_channels = multiscale.channels

        rank = options.format_option("scale")
        rank = 0 if rank is None else rank
        if not (0 <= rank < len(multiscale.dataset_paths)):
            raise ValueError(
                f"scale {rank} does not exist in {path}. Available: "
                f"{list(range(len(multiscale.dataset_paths)))}."
            )
        chosen_dataset_path = multiscale.dataset_paths[rank]
        self._ts_spec = {
            "driver": "zarr3",
            "kvstore": {
                "driver": "zip",
                "base": _make_kvstore(path),
                "path": f"{chosen_dataset_path}/",
            },
        }
        try:
            array = ts.open(
                self._ts_spec, open=True, context=TS_CONTEXT, recheck_cached="open"
            ).result()
        except Exception as e:
            raise CorruptImageError(
                f"Cannot open the OME-Zarr dataset {chosen_dataset_path!r} "
                f"inside {path}. It is likely corrupted or not a valid Zarr "
                "v3 array.",
                path=path,
            ) from e

        domain_labels = list(array.domain.labels)
        axis_name_hint = list(multiscale.axis_names) if multiscale.axis_names else None
        axis_labels = axis_name_hint or (domain_labels if any(domain_labels) else None)
        self._axes = guess_axes(len(array.shape), axis_labels=axis_labels, path=path)

        shape = array.domain.exclusive_max
        axis_to_size = dict(zip(self._axes, shape))
        self._x = axis_to_size.get(X_AXIS, 1)
        self._y = axis_to_size.get(Y_AXIS, 1)
        self._z = axis_to_size.get(Z_AXIS, 1)
        raw_num_channels = axis_to_size.get(C_AXIS, 1)
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

    def get_layer_split_options(self) -> dict[str, list[int]] | None:
        if len(self._possible_layers) == 0:
            return None
        return self._possible_layers

    @property
    def suggested_view_configuration(self) -> LayerViewConfiguration | None:
        return _ome_suggested_view_configuration(
            self._omero_channels, self._channel, self.num_channels
        )

    def layer_split_label(self, key: str, value: int) -> str:
        return _ome_layer_split_label(
            self._omero_channels, key, value
        ) or super().layer_split_label(key, value)
