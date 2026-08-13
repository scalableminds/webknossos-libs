import json
import zipfile
from os import PathLike
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import numpy as np
from upath import UPath

from webknossos.dataset_properties import SEGMENTATION_CATEGORY
from webknossos.geometry import Mag, NDBoundingBox, Vec3Int, Vec3IntLike
from webknossos.utils import WkImportError

from ..defaults import (
    DEFAULT_CHUNK_SHAPE,
    DEFAULT_CHUNKS_PER_SHARD_FROM_IMAGES,
    DEFAULT_SHARD_SHAPE,
    DEFAULT_SHARD_SHAPE_FROM_IMAGES,
)

if TYPE_CHECKING:
    from .abstract_layer import AbstractLayer
    from .view import MagView

# Constants for as_ome_tiff's pyramid levels, ported from the reference
# tiff-export implementation.
OME_TIFF_DOWNSAMPLING_STEPS = 5
OME_TIFF_DOWNSAMPLING_MIN_PIXEL = 16


def _resolve_export_bbox(
    layer: "AbstractLayer", bounding_box: NDBoundingBox | None
) -> NDBoundingBox:
    bbox = bounding_box if bounding_box is not None else layer.bounding_box
    return bbox.intersected_with(layer.bounding_box)


def _resolve_export_mag(layer: "AbstractLayer", mag: Mag | None) -> "MagView":
    return layer.get_mag(mag) if mag is not None else layer.get_finest_mag()


def _make_tiff_name(filename_prefix: str, slice_index: int, digits: int) -> str:
    if not filename_prefix:
        return f"{slice_index:0{digits}d}.tiff"
    return f"{filename_prefix}_{slice_index:0{digits}d}.tiff"


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


def _squeeze_nd_data_to_3d(
    data: np.ndarray,
    bbox: NDBoundingBox,
    mag_view: "MagView",
    is_single_section: bool,
) -> np.ndarray:
    nd_detection_limit = 2 if is_single_section else 3
    has_additional_axes = len(bbox.axes) > nd_detection_limit
    if not has_additional_axes:
        return data
    additional_axes_indices = []
    channel_offset = 1
    axes = mag_view.bounding_box.axes
    if is_single_section:
        # z is already removed from the shape of single-section data.
        axes = tuple(axis for axis in axes if axis != "z")
    for index, axis_name in enumerate(axes):
        if axis_name not in ("x", "y", "z"):
            additional_axes_indices.append(index + channel_offset)
    # All additional axes only extend by one value, so they can be dropped.
    return data.squeeze(axis=tuple(additional_axes_indices))


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
            {"ome": {"version": ome_version}, "jsonFirst": True}
        ).encode("utf-8")


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
    ) -> None:
        """Exports the layer as a single, zipped OME-Zarr archive compliant
        with [NGFF RFC-9](https://ngff.openmicroscopy.org/rfc/9/), "Zipped
        OME-Zarr" (`.ozx`).

        If `bounding_box` is given, the export is cropped to it (intersected
        with the layer's own bounding box). If `mag` is given, the archive
        contains that mag plus every coarser mag already present on the
        layer; if `mag` is None, the full mag pyramid is exported.

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
        bbox = _resolve_export_bbox(layer, bounding_box)
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
                bounding_box=bbox,
                largest_segment_id=layer._get_largest_segment_id_maybe(),
            )
            for target_mag in target_mags:
                source_mag_view = layer.get_mag(target_mag)
                local_bbox = bbox.align_with_mag(target_mag)
                if local_bbox.is_empty():
                    continue
                data = source_mag_view.read(absolute_bounding_box=local_bbox)
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
                    local_size = local_bbox.in_mag(target_mag).size_xyz
                    mag_shard_shape = _shard_shape_for_export(
                        local_size, _default_max_shard_shape(local_size)
                    )
                target_mag_view = tmp_layer.add_mag(
                    target_mag, shard_shape=mag_shard_shape
                )
                target_mag_view.write(
                    data,
                    absolute_bounding_box=local_bbox,
                    allow_resize=True,
                    allow_unaligned=True,
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
        """Exports the layer as a directory of per-slice TIFF files, one file
        per z-section, under `output_path`. Files are named `NNNNNN.tiff`, or
        `{filename_prefix}_NNNNNN.tiff` if `filename_prefix` is given.
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

        with mag_view.get_buffered_slice_reader(absolute_bounding_box=bbox) as reader:
            for slice_index, slice_data in enumerate(reader):
                image = _slice_to_image(
                    _squeeze_nd_data_to_3d(
                        slice_data, bbox, mag_view, is_single_section=True
                    )
                )
                tiff_path = output_path / _make_tiff_name(
                    filename_prefix, slice_index, digits
                )
                with tiff_path.open("wb") as f:
                    tifffile.imwrite(f, data=image, compression=compression)

    def as_ome_tiff(
        self,
        *,
        output_path: str | PathLike | UPath,
        bounding_box: NDBoundingBox | None = None,
        mag: Mag | None = None,
    ) -> None:
        """Exports the layer as a single, pyramidal OME-TIFF file."""
        try:
            import tifffile
        except ImportError as e:
            raise WkImportError("tifffile", "tifffile") from e

        output_path = UPath(output_path)
        layer = self._layer
        mag_view = _resolve_export_mag(layer, mag)
        bbox = _resolve_export_bbox(layer, bounding_box).align_with_mag(mag_view.mag)
        compression = "zlib" if layer.category == SEGMENTATION_CATEGORY else None

        voxel_size_nm = (
            layer.dataset.voxel_size_with_unit.to_nanometer().to_np()
            * mag_view.mag.to_np()
        )

        data = mag_view.read(absolute_bounding_box=bbox)
        data = _squeeze_nd_data_to_3d(data, bbox, mag_view, is_single_section=False)
        data = data.swapaxes(1, 3)  # (C,X,Y,Z) -> (C,Z,Y,X)

        with (
            output_path.open("wb") as f,
            tifffile.TiffWriter(f, bigtiff=True) as tif,
        ):
            metadata = {
                "axes": "CZYX",
                "PhysicalSizeX": voxel_size_nm[0],
                "PhysicalSizeXUnit": "nm",
                "PhysicalSizeY": voxel_size_nm[1],
                "PhysicalSizeYUnit": "nm",
                "PhysicalSizeZ": voxel_size_nm[2],
                "PhysicalSizeZUnit": "nm",
                "Channel": {"Name": [f"Channel {i}" for i in range(data.shape[0])]},
            }

            downsampling_factors = [
                factor
                for factor in (
                    2 ** (step + 1) for step in range(OME_TIFF_DOWNSAMPLING_STEPS)
                )
                if data.shape[2] // factor > OME_TIFF_DOWNSAMPLING_MIN_PIXEL
                and data.shape[3] // factor > OME_TIFF_DOWNSAMPLING_MIN_PIXEL
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
                tif.write(
                    data[:, :, ::factor, ::factor],  # cheap downsampling
                    subfiletype=1,
                    resolution=(
                        (factor * voxel_size_nm[0]) / 1e7,
                        (factor * voxel_size_nm[1]) / 1e7,
                    ),
                    resolutionunit="CENTIMETER",
                    compression=compression,
                )
