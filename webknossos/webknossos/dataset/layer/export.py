import itertools
import json
import zipfile
from os import PathLike
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

import numpy as np
from cluster_tools import Executor
from upath import UPath

from webknossos.dataset_properties import SEGMENTATION_CATEGORY
from webknossos.geometry import (
    BoundingBox,
    Mag,
    NDBoundingBox,
    Vec3Int,
    Vec3IntLike,
    VecInt,
)
from webknossos.utils import WkImportError, wait_and_ensure_success, wrap_executor

from ..defaults import (
    DEFAULT_CHUNK_SHAPE,
    DEFAULT_CHUNKS_PER_SHARD_FROM_IMAGES,
    DEFAULT_SHARD_SHAPE,
    DEFAULT_SHARD_SHAPE_FROM_IMAGES,
)

if TYPE_CHECKING:
    from .abstract_layer import AbstractLayer
    from .view import MagView, View

# Constants for as_ome_tiff's pyramid levels, ported from the reference
# tiff-export implementation.
OME_TIFF_DOWNSAMPLING_STEPS = 5
OME_TIFF_DOWNSAMPLING_MIN_PIXEL = 16

# OME-TIFF only supports this fixed set of axes.
_OME_TIFF_AXIS_CODES = {"c": "C", "t": "T", "z": "Z", "y": "Y", "x": "X"}


def _resolve_export_bbox(
    layer: "AbstractLayer", bounding_box: NDBoundingBox | None
) -> NDBoundingBox:
    return (
        layer.bounding_box
        if bounding_box is None
        else bounding_box.intersected_with(layer.bounding_box)
    )


def _resolve_export_mag(layer: "AbstractLayer", mag: Mag | None) -> "MagView":
    return layer.get_mag(mag) if mag is not None else layer.get_finest_mag()


def _data_axes(bbox: NDBoundingBox) -> tuple[str, ...]:
    """The axis order `mag_view.read(absolute_bounding_box=bbox)` actually
    returns. A plain `BoundingBox` (x,y,z only) always gets a channel axis
    prepended; a general `NDBoundingBox` only has "c" if it was already
    present in `bbox.axes` - mirrors `NDBoundingBox`/`BoundingBox.normalize_axes()`.
    """
    if isinstance(bbox, BoundingBox):
        return ("c", *bbox.axes)
    return bbox.axes


def _make_tiff_name(filename_prefix: str, slice_index: int, digits: int) -> str:
    if not filename_prefix:
        return f"{slice_index:0{digits}d}.tiff"
    return f"{filename_prefix}_{slice_index:0{digits}d}.tiff"


def _make_tiff_stack_name(
    filename_prefix: str,
    extra_axes: list[str],
    combo: tuple[int, ...],
    extra_digits: dict[str, int],
    slice_index: int,
    z_digits: int,
) -> str:
    segments = [
        f"{axis}{i:0{extra_digits[axis]}d}" for axis, i in zip(extra_axes, combo)
    ]
    segments.append(f"z{slice_index:0{z_digits}d}")
    body = "_".join(segments)
    return f"{filename_prefix}_{body}.tiff" if filename_prefix else f"{body}.tiff"


def _extract_tiff_stack_slice(
    slice_data: np.ndarray, slice_axes: tuple[str, ...]
) -> np.ndarray:
    """Squeezes the (size-1) extra axes out of a per-slice array yielded by
    `get_buffered_slice_reader` (whose axis order is `slice_axes`), and
    reorders the rest to (c, x, y) - synthesizing a channel axis if the
    layer doesn't have one, so `_slice_to_image` always sees (C, X, Y).
    """
    extra = [a for a in slice_axes if a not in ("c", "x", "y")]
    squeezed = np.squeeze(slice_data, axis=tuple(slice_axes.index(a) for a in extra))
    remaining = tuple(a for a in slice_axes if a not in extra)
    target = tuple(a for a in ("c", "x", "y") if a in remaining)
    reordered = np.moveaxis(
        squeezed, [remaining.index(a) for a in target], list(range(len(target)))
    )
    if "c" not in target:
        reordered = np.expand_dims(reordered, axis=0)
    return reordered


def _ome_tiff_axes_and_permutation(
    bbox: NDBoundingBox,
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """The OME-TIFF axis order (y, x always last, as tifffile requires) and
    the permutation of `mag_view.read()`'s axes needed to reach it.
    """
    data_axes = _data_axes(bbox)
    unsupported = [a for a in data_axes if a not in _OME_TIFF_AXIS_CODES]
    if unsupported:
        raise ValueError(
            f"as_ome_tiff does not support axis/axes {unsupported!r}; "
            "OME-TIFF only supports c, t, z, y, x."
        )
    other = [a for a in data_axes if a not in ("x", "y")]
    canonical = (*other, "y", "x")
    return canonical, tuple(data_axes.index(a) for a in canonical)


def _default_max_shard_shape(local_size: Vec3Int) -> Vec3Int:
    """The cap used by `_shard_shape_for_export` when no explicit
    `shard_shape` is given: for flat data (few z sections, e.g. an exported
    2D crop), the wider from-images default, otherwise the regular default
    shard shape. Mirrors `Layer.add_mag`'s own default selection.
    """
    if local_size.z <= DEFAULT_CHUNK_SHAPE.z * DEFAULT_CHUNKS_PER_SHARD_FROM_IMAGES.z:
        return DEFAULT_SHARD_SHAPE_FROM_IMAGES
    return DEFAULT_SHARD_SHAPE


def _shard_shape_for_export(local_size: Vec3Int, max_shard_shape: Vec3Int) -> Vec3Int:
    """Picks a power-of-two shard shape (a multiple of the default chunk
    shape) that covers `local_size` in a single shard per axis, capped at
    `max_shard_shape` so a large export doesn't fragment into more, smaller
    shards than necessary.
    """

    def axis_shard(size: int, chunk: int, max_shard: int) -> int:
        shard = chunk
        while shard < size and shard < max_shard:
            shard *= 2
        return min(shard, max_shard)

    return Vec3Int(
        axis_shard(local_size.x, DEFAULT_CHUNK_SHAPE.x, max_shard_shape.x),
        axis_shard(local_size.y, DEFAULT_CHUNK_SHAPE.y, max_shard_shape.y),
        axis_shard(local_size.z, DEFAULT_CHUNK_SHAPE.z, max_shard_shape.z),
    )


def _slice_to_image(data_slice: np.ndarray) -> np.ndarray:
    if data_slice.shape[0] == 1:
        # discard greyscale dimension
        data_slice = data_slice.squeeze(axis=0)
        # swap the axis
        data_slice = data_slice.transpose((1, 0))
    else:
        # swap axis and move the channel axis
        data_slice = data_slice.transpose((2, 1, 0))
    return data_slice


def _write_ome_zarr_zip(layer_dir: UPath, output_path: UPath, ome_version: str) -> None:
    """Packages an OME-Zarr (Zarr v3) layer directory into an RFC-9 compliant
    zip file, see https://ngff.openmicroscopy.org/rfc/9/.
    """
    # All zarr.json metadata files must come first, in breadth-first order,
    # so that a reader can find the whole hierarchy's metadata without
    # scanning the (potentially huge) list of chunk/shard entries.
    metadata_files = sorted(
        layer_dir.rglob("zarr.json"),
        key=lambda p: (len(p.relative_to(layer_dir).parts), p.as_posix()),
    )
    data_files = sorted(
        p for p in layer_dir.rglob("*") if p.is_file() and p.name != "zarr.json"
    )

    with (
        output_path.open("wb") as out,
        zipfile.ZipFile(out, mode="w") as zip_file,
    ):
        for file_path in [*metadata_files, *data_files]:
            arcname = file_path.relative_to(layer_dir).as_posix()
            zip_info = zipfile.ZipInfo.from_file(str(file_path), arcname)
            # Zarr chunks are already compressed, and RFC-9 recommends
            # disabling zip-level compression in favor of the Zarr codecs.
            zip_info.compress_type = zipfile.ZIP_STORED
            with (
                file_path.open("rb") as source,
                # force_zip64 ensures the ZIP64 extension is always used,
                # as recommended by RFC-9, irrespective of the archive size.
                zip_file.open(zip_info, mode="w", force_zip64=True) as dest,
            ):
                dest.write(source.read())

        zip_file.comment = json.dumps(
            {
                "ome": {"version": ome_version},
                "jsonFirst": True,
            }
        ).encode("utf-8")


def _copy(args: "tuple[View, MagView, NDBoundingBox]") -> None:
    source, target_mag_view, chunk_bbox = args
    # chunk_bbox is aligned to the target mag's own (x,y,z-only) shard grid,
    # so this is safe for plain 3D layers without allow_unaligned. It's only
    # needed for layers with additional axes (e.g. time): those are always
    # chunked to size 1 (see as_ozx), so a write covering a full x/y/z shard
    # but a single value along such an axis still lands entirely within its
    # own, fully owned shard file, but WEBKNOSSOS's shard-alignment check
    # only looks at x/y/z and flags it as unaligned regardless.
    target_mag_view.write(
        source.read(), absolute_bounding_box=chunk_bbox, allow_unaligned=True
    )


class LayerExport:
    """Exports a layer's data to common bioimaging file formats.

    Accessible via `layer.export`, e.g.:
    ```python
    layer.export.as_ozx(output_path=UPath("out.ozx"))
    layer.export.as_tiff_stack(output_path=UPath("out_tiffs"))
    layer.export.as_ome_tiff(output_path=UPath("out.ome.tif"))
    ```
    """

    def __init__(self, layer: "AbstractLayer") -> None:
        self._layer = layer

    def as_ozx(
        self,
        *,
        output_path: str | PathLike | UPath,
        bounding_box: NDBoundingBox | None = None,
        mag: Mag | None = None,
        shard_shape: Vec3IntLike | int | None = None,
        executor: Executor | None = None,
    ) -> None:
        """Exports the layer as a single, zipped OME-Zarr archive compliant
        with [NGFF RFC-9](https://ngff.openmicroscopy.org/rfc/9/), "Zipped
        OME-Zarr" (`.ozx`). Layers with additional axes (e.g. time) are
        supported - the OME axes/scale metadata is derived from the
        layer's own axes.

        If `bounding_box` is given, the export is cropped to it (intersected
        with the layer's own bounding box). If `mag` is given, the archive
        contains that mag plus every coarser mag already present on the
        layer; if `mag` is None, the full mag pyramid is exported.

        The output bounding box is translated to origin.

        `shard_shape` fixes the shard shape used for every exported mag. If
        omitted, a shard shape is picked per mag that just covers the
        exported region (searching upward from the default chunk shape),
        capped at the default shard shape (or the wider from-images default
        for flat data with few z sections).
        """
        output_path = UPath(output_path)
        fixed_shard_shape = (
            Vec3Int.from_vec_or_int(shard_shape) if shard_shape is not None else None
        )
        layer = self._layer
        source_bbox = _resolve_export_bbox(layer, bounding_box)
        target_bbox = source_bbox.with_topleft(VecInt.zeros(axes=source_bbox.axes))
        target_mags = sorted(m for m in layer.mags if mag is None or m >= mag)

        with TemporaryDirectory() as tmpdir:
            from ..dataset import Dataset  # local import avoids a circular import

            tmp_dataset = Dataset(
                dataset_path=tmpdir,
                voxel_size_with_unit=layer.dataset.voxel_size_with_unit,
            )
            # RFC-9 requires Zarr v3, regardless of the source layer's format.
            tmp_layer = tmp_dataset.add_layer(
                layer_name=layer.name,
                category=layer.category,
                dtype=layer.dtype,
                num_channels=layer.num_channels,
                data_format="zarr3",
                bounding_box=target_bbox,
                largest_segment_id=layer._get_largest_segment_id_maybe(),
            )
            for target_mag in target_mags:
                source_mag_view = layer.get_mag(target_mag)
                source_local_bbox = source_bbox.align_with_mag(target_mag)
                target_local_bbox = target_bbox.align_with_mag(target_mag)
                if target_local_bbox.is_empty():
                    continue
                if fixed_shard_shape is not None:
                    mag_shard_shape = fixed_shard_shape
                else:
                    # webknossos Zarr arrays always span from voxel 0, so a
                    # large fixed shard shape would pad a small crop's
                    # declared extent far beyond what was actually exported.
                    # Instead, size each shard to just cover the exported
                    # region per axis (one shard, if possible), capped at a
                    # sensible default so a large export doesn't fragment
                    # into more, smaller shards than necessary.
                    local_size = target_local_bbox.in_mag(target_mag).size_xyz
                    mag_shard_shape = _shard_shape_for_export(
                        local_size, _default_max_shard_shape(local_size)
                    )
                target_mag_view = tmp_layer.add_mag(
                    target_mag, shard_shape=mag_shard_shape
                )
                # target_local_bbox is in Mag(1) units, so the shard shape
                # (in this mag's own voxel units) needs to be scaled up to
                # Mag(1) before it can be used to chunk the bbox.
                shard_shape_mag1 = mag_shard_shape * target_mag.to_vec3_int()
                with wrap_executor(executor) as mag_executor:
                    jobs = [
                        (
                            source_mag_view.get_view(
                                absolute_bounding_box=chunk_bbox.with_topleft(
                                    chunk_bbox.topleft + source_local_bbox.topleft
                                ),
                                read_only=True,
                            ),
                            target_mag_view,
                            chunk_bbox,
                        )
                        for chunk_bbox in target_local_bbox.chunk(
                            shard_shape_mag1, shard_shape_mag1
                        )
                    ]
                    wait_and_ensure_success(
                        mag_executor.map_to_futures(_copy, jobs),
                        executor=mag_executor,
                        progress_desc=f"Exporting {target_mag}",
                    )

            _write_ome_zarr_zip(tmp_layer.path, output_path, ome_version="0.5")

    def as_tiff_stack(
        self,
        *,
        output_path: str | PathLike | UPath,
        bounding_box: NDBoundingBox | None = None,
        mag: Mag | None = None,
        filename_prefix: str = "",
    ) -> None:
        """Exports the layer as a directory of per-slice TIFF files, one
        file per z-section (and per combination of any additional axes,
        e.g. time), under `output_path`.

        Files are named `NNNNNN.tiff` (or `{filename_prefix}_NNNNNN.tiff`)
        for plain 3D layers, or `{axis}NNN_..._zNNN.tiff` (one segment per
        additional axis, sorted by name, plus z; still prefixed with
        `filename_prefix` if given) for layers with additional axes.
        """
        try:
            import tifffile
        except ImportError as e:
            raise WkImportError("tifffile", "tifffile") from e

        output_path = UPath(output_path)
        layer = self._layer
        mag_view = _resolve_export_mag(layer, mag)
        bbox = _resolve_export_bbox(layer, bounding_box).align_with_mag(mag_view.mag)

        output_path.mkdir(parents=True, exist_ok=True)
        compression = "zlib" if layer.category == SEGMENTATION_CATEGORY else None

        num_slices = bbox.in_mag(mag_view.mag).size_xyz.z
        digits = max(1, len(str(max(num_slices - 1, 0))))

        extra_axes = sorted(a for a in bbox.axes if a not in ("c", "x", "y", "z"))
        extra_digits = {
            a: max(1, len(str(max(bbox.get_shape(a) - 1, 0)))) for a in extra_axes
        }

        for combo in itertools.product(*(range(bbox.get_shape(a)) for a in extra_axes)):
            sub_bbox = bbox
            for axis, i in zip(extra_axes, combo):
                sub_bbox = sub_bbox.with_bounds(axis, bbox.topleft[axis] + i, 1)
            slice_axes = tuple(a for a in _data_axes(sub_bbox) if a != "z")

            with mag_view.get_buffered_slice_reader(
                absolute_bounding_box=sub_bbox
            ) as reader:
                for slice_index, slice_data in enumerate(reader):
                    image = _slice_to_image(
                        _extract_tiff_stack_slice(slice_data, slice_axes)
                    )
                    name = (
                        _make_tiff_stack_name(
                            filename_prefix,
                            extra_axes,
                            combo,
                            extra_digits,
                            slice_index,
                            digits,
                        )
                        if extra_axes
                        else _make_tiff_name(filename_prefix, slice_index, digits)
                    )
                    tiff_path = output_path / name
                    with tiff_path.open("wb") as f:
                        tifffile.imwrite(f, data=image, compression=compression)

    def as_ome_tiff(
        self,
        *,
        output_path: str | PathLike | UPath,
        bounding_box: NDBoundingBox | None = None,
        mag: Mag | None = None,
    ) -> None:
        """Exports the layer as a single, pyramidal OME-TIFF file.

        Only layers whose axes are a subset of c, t, z, y, x are supported
        (OME-TIFF's own dimension model doesn't extend to other axes).
        """
        try:
            import tifffile
        except ImportError as e:
            raise WkImportError("tifffile", "tifffile") from e

        output_path = UPath(output_path)
        layer = self._layer
        mag_view = _resolve_export_mag(layer, mag)
        bbox = _resolve_export_bbox(layer, bounding_box).align_with_mag(mag_view.mag)
        compression = "zlib" if layer.category == SEGMENTATION_CATEGORY else None

        canonical_axes, permutation = _ome_tiff_axes_and_permutation(bbox)
        axes_str = "".join(_OME_TIFF_AXIS_CODES[a] for a in canonical_axes)
        x_idx = axes_str.index("X")
        y_idx = axes_str.index("Y")

        voxel_size_nm = (
            layer.dataset.voxel_size_with_unit.to_nanometer().to_np()
            * mag_view.mag.to_np()
        )

        data = mag_view.read(absolute_bounding_box=bbox)
        data = np.transpose(data, permutation)

        with (
            output_path.open("wb") as f,
            tifffile.TiffWriter(f, bigtiff=True) as tif,
        ):
            metadata: dict[str, Any] = {
                "axes": axes_str,
                "PhysicalSizeX": voxel_size_nm[0],
                "PhysicalSizeXUnit": "nm",
                "PhysicalSizeY": voxel_size_nm[1],
                "PhysicalSizeYUnit": "nm",
                "PhysicalSizeZ": voxel_size_nm[2],
                "PhysicalSizeZUnit": "nm",
            }
            if "c" in canonical_axes:
                metadata["Channel"] = {
                    "Name": [f"Channel {i}" for i in range(layer.num_channels)]
                }

            downsampling_factors = [
                factor
                for factor in (
                    2 ** (step + 1) for step in range(OME_TIFF_DOWNSAMPLING_STEPS)
                )
                if data.shape[y_idx] // factor > OME_TIFF_DOWNSAMPLING_MIN_PIXEL
                and data.shape[x_idx] // factor > OME_TIFF_DOWNSAMPLING_MIN_PIXEL
            ]

            tif.write(
                data,
                subifds=len(downsampling_factors),
                resolution=(voxel_size_nm[0] / 1e7, voxel_size_nm[1] / 1e7),
                metadata=metadata,
                resolutionunit="CENTIMETER",
                compression=compression,
            )
            for factor in downsampling_factors:
                strides = tuple(
                    slice(None, None, factor) if i in (x_idx, y_idx) else slice(None)
                    for i in range(data.ndim)
                )
                tif.write(
                    data[strides],  # cheap downsampling
                    subfiletype=1,
                    resolution=(
                        (factor * voxel_size_nm[0]) / 1e7,
                        (factor * voxel_size_nm[1]) / 1e7,
                    ),
                    resolutionunit="CENTIMETER",
                    compression=compression,
                )
