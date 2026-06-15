import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import numpy as np
from cluster_tools import Executor

from webknossos.geometry import BoundingBox, Mag, NDBoundingBox, Vec3Int, Vec3IntLike
from webknossos.geometry.mag import MagLike
from webknossos.utils import named_partial, wrap_executor

from .view import MagView, View

if TYPE_CHECKING:
    from webknossos.dataset.layer.layer import Layer

logger = logging.getLogger(__name__)

# Maps an (N, 3) array of Mag(1) coordinates to an (N, 3) array of Mag(1)
# coordinates. Rows may contain NaN to mark coordinates without a source.
TransformFunction = Callable[[np.ndarray], np.ndarray]

# Number of threads per job used to process buffer tiles. The tile reads are
# IO-bound and the coordinate math releases the GIL, so a few threads hide the
# read latency. Kept small because the storage backend (e.g. tensorstore) adds
# its own read concurrency and jobs typically already run one per core.
_NUM_TILE_THREADS = 4


def determine_transform_buffer_shape(input_chunk_shape: Vec3Int) -> Vec3Int:
    # Roughly 128**3 voxels per processing tile (bounding the coordinate arrays
    # to a few hundred MB), rounded up to a multiple of the input storage chunk
    # shape since reads decompress whole input chunks anyway.
    return input_chunk_shape * Vec3Int.full(128).ceildiv(input_chunk_shape).pairmax(1)


class AffineTransform:
    """A picklable callable that applies a 4x4 homogeneous affine matrix to (N, 3) point arrays."""

    def __init__(self, matrix: np.ndarray) -> None:
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError(
                f"The affine matrix must have shape (4, 4), got {matrix.shape}."
            )
        self.matrix = matrix

    def __call__(self, points: np.ndarray) -> np.ndarray:
        return points @ self.matrix[:3, :3].T + self.matrix[:3, 3]

    def inverse(self) -> "AffineTransform":
        return AffineTransform(np.linalg.inv(self.matrix))

    def transform_bbox(self, bbox: BoundingBox) -> BoundingBox:
        corners = np.array(
            [
                (x, y, z)
                for x in (bbox.topleft.x, bbox.bottomright.x)
                for y in (bbox.topleft.y, bbox.bottomright.y)
                for z in (bbox.topleft.z, bbox.bottomright.z)
            ],
            dtype=np.float64,
        )
        transformed = self(corners)
        topleft = np.floor(transformed.min(axis=0)).astype(np.int64)
        bottomright = np.ceil(transformed.max(axis=0)).astype(np.int64)
        return BoundingBox(topleft=topleft, size=bottomright - topleft)


def _transform_tile(
    tile_bbox: NDBoundingBox,
    output_data: np.ndarray,
    chunk_topleft: np.ndarray,
    input_mag_view: MagView,
    input_mask_mag_view: MagView | None,
    inverse_transform: TransformFunction,
    translation: Vec3Int,
    input_bbox: BoundingBox,
    mag: Mag,
) -> None:
    """Fills the part of `output_data` (the buffer of the whole chunk) covered by `tile_bbox`.

    Tiles write disjoint regions of `output_data`, so this is safe to run in
    multiple threads without locking.
    """
    mag_np = mag.to_np()
    topleft = tile_bbox.topleft.to_np()
    bottomright = tile_bbox.bottomright.to_np()

    # Build the grid of all output voxel positions of this tile (in Mag(1) coordinates)
    grid = np.meshgrid(
        np.arange(topleft[0], bottomright[0], step=mag_np[0], dtype=np.float64),
        np.arange(topleft[1], bottomright[1], step=mag_np[1], dtype=np.float64),
        np.arange(topleft[2], bottomright[2], step=mag_np[2], dtype=np.float64),
        indexing="ij",
    )
    output_coords = np.stack([g.flatten() for g in grid], axis=1)
    del grid
    input_coords = np.asarray(
        inverse_transform(output_coords - translation.to_np()), dtype=np.float64
    )

    # NaN rows mark output voxels without a source; additionally only voxels
    # mapping into the input layer's bounding box can be filled. Coordinates are
    # valid up to the last sample position (bottomright - mag) per axis, matching
    # scipy.ndimage.affine_transform with order=0 and mode="constant".
    with np.errstate(invalid="ignore"):
        in_bounds = (
            ~np.isnan(input_coords).any(axis=1)
            & (input_coords >= np.array(input_bbox.topleft)).all(axis=1)
            & (input_coords <= np.array(input_bbox.bottomright) - mag_np).all(axis=1)
        )
    input_coords = input_coords[in_bounds]
    if len(input_coords) == 0:
        return  # nothing maps into this tile, the buffer keeps its initial values
    output_coords = output_coords[in_bounds]

    # The 2 * mag margin guarantees that the rounded relative input coordinates
    # below stay within the read buffer.
    input_topleft = np.floor(input_coords.min(axis=0)).astype(np.int64)
    relevant_input_bbox = BoundingBox(
        topleft=input_topleft,
        size=np.round(input_coords.max(axis=0) + 2 * mag_np - input_topleft).astype(
            np.int64
        ),
    ).align_with_mag(mag, ceil=True)

    input_data = input_mag_view.read(absolute_bounding_box=relevant_input_bbox)

    output_coords = ((output_coords - chunk_topleft) / mag_np).astype(np.int64)
    # Nearest-neighbor with ties rounding up (floor(x + 0.5)), matching scipy.ndimage
    input_coords = np.floor(
        (input_coords - np.array(relevant_input_bbox.topleft)) / mag_np + 0.5
    ).astype(np.int64)

    if input_mask_mag_view is not None:
        mask_data = input_mask_mag_view.read(absolute_bounding_box=relevant_input_bbox)
        keep = (
            mask_data[:, input_coords[:, 0], input_coords[:, 1], input_coords[:, 2]] > 0
        ).all(axis=0)
        output_coords = output_coords[keep]
        input_coords = input_coords[keep]
        if len(input_coords) == 0:
            return

    output_data[:, output_coords[:, 0], output_coords[:, 1], output_coords[:, 2]] = (
        input_data[:, input_coords[:, 0], input_coords[:, 1], input_coords[:, 2]]
    )


def transform_chunk_job(
    args: tuple[View, int],
    input_mag_view: MagView,
    input_mask_mag_view: MagView | None,
    inverse_transform: TransformFunction,
    translation: Vec3Int,
    input_bbox: BoundingBox,
    mag: Mag,
    buffer_shape: Vec3Int,
    fill_value: int | float | None,
) -> None:
    (output_view, _i) = args
    try:
        chunk_bbox = output_view.bounding_box
        output_shape = chunk_bbox.in_mag(mag).size
        if fill_value is None:
            # Initialize the buffer with the existing output data so that voxels
            # without a source (out-of-bounds, NaN or masked out) keep their value.
            output_data = output_view.read()
        else:
            output_data = np.full(
                (input_mag_view.layer.num_channels, *output_shape),
                fill_value,
                dtype=input_mag_view.get_dtype(),
            )

        # Process the chunk in smaller tiles to bound the size of the coordinate
        # arrays and of the input reads, overlapping IO with a few threads.
        buffer_shape_mag1 = buffer_shape * mag.to_vec3_int()
        tile_bboxes = list(chunk_bbox.chunk(buffer_shape_mag1, buffer_shape_mag1))
        with ThreadPoolExecutor(
            max_workers=min(_NUM_TILE_THREADS, len(tile_bboxes))
        ) as pool:
            for _ in pool.map(
                lambda tile_bbox: _transform_tile(
                    tile_bbox,
                    output_data,
                    chunk_bbox.topleft.to_np(),
                    input_mag_view,
                    input_mask_mag_view,
                    inverse_transform,
                    translation,
                    input_bbox,
                    mag,
                ),
                tile_bboxes,
            ):
                pass

        output_view.write(output_data)
    except Exception as exc:
        logger.exception(
            f"Transforming chunk {output_view.bounding_box} failed with {exc}"
        )
        raise exc


def transform(
    input_layer: "Layer",
    output_layer: "Layer",
    inverse_transform: Callable[[np.ndarray], np.ndarray],
    *,
    mag: MagLike | None = None,
    output_bbox: NDBoundingBox | None = None,
    translate_to_positive: bool = True,
    input_mask_layer: "Layer | None" = None,
    fill_value: int | float | None = None,
    chunk_shape: Vec3IntLike | None = None,
    buffer_shape: Vec3IntLike | int | None = None,
    executor: Executor | None = None,
    progress_desc: str | None = None,
) -> BoundingBox:
    """Resamples `input_layer`'s data into `output_layer` using an arbitrary coordinate transform.

    For every voxel position of the output bounding box, the `inverse_transform` is used
    to look up the corresponding position in `input_layer` and the data is copied using
    nearest-neighbor sampling. Output voxels without a source (mapping outside the input
    layer's bounding box, transformed coordinates containing NaN, or masked out by
    `input_mask_layer`) keep their previous value, or are set to `fill_value` if it is
    given. The sampling conventions match `scipy.ndimage.affine_transform` with
    `order=0` and `mode="constant"`: rounding ties are rounded up and coordinates are
    considered in-bounds up to the last sample position (`bottomright - mag`) per axis.

    Args:
        input_layer: Layer to read data from.
        output_layer: Existing layer to write the transformed data to. Must have the same
            dtype and number of channels as `input_layer`. May belong to a different dataset.
        inverse_transform: Callable mapping an (N, 3) array of output positions to an
            (N, 3) array of input positions, both in Mag(1) coordinates. Rows may contain
            NaN to mark positions without a source. Must be picklable (e.g. a module-level
            function, a functools.partial of one, or a callable class instance) because
            the default executor and most explicit executors are process-based.
        mag: Magnification to transform. Defaults to the finest available mag of `input_layer`.
        output_bbox: Region of the output layer to fill, in Mag(1) coordinates. Defaults to
            the output layer's current bounding box.
        translate_to_positive: If True, an output bounding box with negative topleft is
            shifted into positive space (and `inverse_transform` inputs are shifted back
            accordingly). If False, a negative topleft raises a ValueError.
        input_mask_layer: Optional mask layer (same dataset geometry as `input_layer`). Only
            voxels where the mask is greater than zero in all channels are copied.
        fill_value: Value for output voxels without a source. If None (default), the
            existing output data is read first and such voxels keep their previous
            value. If set, the output buffer is initialized with this value instead,
            so the whole output bounding box is overwritten deterministically.
        chunk_shape: Size of the chunks to process per job, in Mag(1) coordinates. Must be
            a multiple of the output mag's shard shape (in Mag(1)), so that parallel jobs
            never write to the same shard. Defaults to one shard per job.
        buffer_shape: Size of the processing tiles within a chunk, in voxels of the
            current mag. Bounds the memory per job: the coordinate arrays take roughly
            100 bytes per buffer voxel per tile thread. Defaults to roughly 128**3
            voxels, rounded up to a multiple of the input mag's storage chunk shape.
        executor: Executor for parallel processing (e.g. multiprocessing, slurm). If None,
            a multiprocessing executor is used.
        progress_desc: Description for the progress bar.

    Returns:
        BoundingBox: The bounding box that was written to the output layer (mag-aligned
            and, if applicable, shifted by the translate_to_positive translation).

    Raises:
        KeyError: If the mag does not exist in `input_layer` or in `input_mask_layer`.
        ValueError: If layer properties are incompatible, the output bounding box is empty,
            or it has a negative topleft while translate_to_positive is False.

    Examples:
        ```python
        def shift_by_100(points: np.ndarray) -> np.ndarray:
            return points - 100  # output voxel x maps to input voxel x - 100

        output_layer = output_dataset.add_layer_like(layer, layer.name)
        transform(
            layer,
            output_layer,
            shift_by_100,
            output_bbox=layer.bounding_box.offset((100, 100, 100)),
        )
        ```

    Note:
        - Sampling is nearest-neighbor, no interpolation is performed.
        - Each chunk is processed in `buffer_shape`-sized tiles using a few threads to
          overlap reads with computation. Each tile reads the axis-aligned bounding box
          of its transformed extent from `input_layer`, which can be considerably larger
          than the tile itself for rotations.
    """
    output_layer._dataset._ensure_writable()

    if not isinstance(input_layer.bounding_box, BoundingBox):
        raise ValueError(
            "transform is only supported for layers with 3D (x, y, z) bounding boxes, "
            + f"got {type(input_layer.bounding_box).__name__} for layer {input_layer.name}."
        )
    if input_layer.bounding_box.is_empty():
        raise ValueError(
            f"The input layer {input_layer.name} has an empty bounding box. "
            + "Please write some data to it or set its bounding box first."
        )
    if input_layer.dtype != output_layer.dtype:
        raise ValueError(
            f"The dtype of the output layer ({output_layer.dtype}) "
            + f"does not match the input layer's dtype ({input_layer.dtype})."
        )
    if input_layer.num_channels != output_layer.num_channels:
        raise ValueError(
            f"The number of channels of the output layer ({output_layer.num_channels}) "
            + f"does not match the input layer's number of channels ({input_layer.num_channels})."
        )

    if mag is None:
        mag = input_layer.get_finest_mag().mag
    else:
        mag = Mag(mag)
        if mag not in input_layer.mags:
            raise KeyError(
                f"Failed to transform layer {input_layer.name}: mag {mag} does not exist."
            )
    if input_mask_layer is not None and mag not in input_mask_layer.mags:
        raise KeyError(
            f"Failed to transform layer {input_layer.name}: mag {mag} does not exist "
            + f"in input_mask_layer {input_mask_layer.name}."
        )

    if output_bbox is None:
        output_bbox = output_layer.bounding_box
    if not isinstance(output_bbox, BoundingBox):
        raise ValueError(
            "transform is only supported for 3D (x, y, z) output bounding boxes, "
            + f"got {type(output_bbox).__name__}."
        )
    if output_bbox.is_empty():
        raise ValueError(
            "The output bounding box is empty. Pass a non-empty output_bbox or set the "
            + "bounding box of the output layer."
        )

    if translate_to_positive:
        translation = Vec3Int(
            np.maximum(-np.array(output_bbox.topleft), 0).astype(np.int64)
        )
        output_bbox = output_bbox.offset(translation)
    else:
        translation = Vec3Int.zeros()
        if not output_bbox.topleft.is_positive():
            raise ValueError(
                f"The output bounding box {output_bbox} has a negative topleft. "
                + "Pass translate_to_positive=True to shift it into positive space."
            )
    output_bbox = output_bbox.align_with_mag(mag, ceil=True)

    # Extending the output layer's bounding box also resizes the mag arrays,
    # which makes the writes below legal.
    if output_layer.bounding_box.is_empty():
        output_layer.bounding_box = output_bbox
    else:
        output_layer.bounding_box = output_layer.bounding_box.extended_by(output_bbox)

    if mag in output_layer.mags:
        output_mag_view = output_layer.get_mag(mag)
    else:
        output_mag_view = output_layer._initialize_mag_from_other_mag(
            mag, input_layer.get_mag(mag), compress=True
        )

    if buffer_shape is None:
        buffer_shape = determine_transform_buffer_shape(
            input_layer.get_mag(mag).info.chunk_shape
        )
    else:
        buffer_shape = Vec3Int.from_vec_or_int(buffer_shape)

    output_view = output_mag_view.get_view(absolute_bounding_box=output_bbox)
    func = named_partial(
        transform_chunk_job,
        input_mag_view=input_layer.get_mag(mag),
        input_mask_mag_view=(
            input_mask_layer.get_mag(mag) if input_mask_layer is not None else None
        ),
        inverse_transform=inverse_transform,
        translation=translation,
        input_bbox=input_layer.bounding_box.align_with_mag(mag, ceil=False),
        mag=mag,
        buffer_shape=buffer_shape,
        fill_value=fill_value,
    )
    if progress_desc is None:
        progress_desc = (
            f"Transforming layer {input_layer.name} into layer {output_layer.name}"
        )
    with wrap_executor(executor) as actual_executor:
        # The default chunk_shape is one shard per job and explicit chunk_shapes are
        # validated to be multiples of the shard shape. Since chunk borders are aligned
        # with absolute multiples of chunk_shape, parallel jobs never share a shard.
        output_view.for_each_chunk(
            func,
            chunk_shape=chunk_shape,
            executor=actual_executor,
            progress_desc=progress_desc,
        )
    return output_bbox


def transform_affine(
    input_layer: "Layer",
    output_layer: "Layer",
    affine_matrix: np.ndarray,
    *,
    mag: MagLike | None = None,
    output_bbox: NDBoundingBox | None = None,
    translate_to_positive: bool = True,
    input_mask_layer: "Layer | None" = None,
    fill_value: int | float | None = None,
    chunk_shape: Vec3IntLike | None = None,
    buffer_shape: Vec3IntLike | int | None = None,
    executor: Executor | None = None,
    progress_desc: str | None = None,
) -> BoundingBox:
    """Resamples `input_layer`'s data into `output_layer` using an affine transformation.

    Wrapper around `transform` for affine transformations. The matrix describes the
    forward transformation from input to output positions; it is inverted internally.

    Args:
        input_layer: Layer to read data from.
        output_layer: Existing layer to write the transformed data to. Must have the same
            dtype and number of channels as `input_layer`. May belong to a different dataset.
        affine_matrix: 4x4 homogeneous matrix describing the forward transformation in
            Mag(1) coordinates. Must be invertible.
        mag: Magnification to transform. Defaults to the finest available mag of `input_layer`.
        output_bbox: Region of the output layer to fill, in Mag(1) coordinates. Defaults to
            the bounding box of `input_layer`'s bounding box transformed with the affine matrix.
        translate_to_positive: If True, an output bounding box with negative topleft is
            shifted into positive space. If False, a negative topleft raises a ValueError.
        input_mask_layer: Optional mask layer (same dataset geometry as `input_layer`). Only
            voxels where the mask is greater than zero in all channels are copied.
        fill_value: Value for output voxels without a source. If None (default), such
            voxels keep their previous value; if set, they are set to this value.
        chunk_shape: Size of the chunks to process per job, in Mag(1) coordinates. Must be
            a multiple of the output mag's shard shape (in Mag(1)). Defaults to one shard
            per job.
        buffer_shape: Size of the processing tiles within a chunk, in voxels of the
            current mag. Defaults to roughly 128**3 voxels, rounded up to a multiple of
            the input mag's storage chunk shape.
        executor: Executor for parallel processing (e.g. multiprocessing, slurm). If None,
            a multiprocessing executor is used.
        progress_desc: Description for the progress bar.

    Returns:
        BoundingBox: The bounding box that was written to the output layer (mag-aligned
            and, if applicable, shifted by the translate_to_positive translation).

    Examples:
        ```python
        scale_by_2 = np.diag([2.0, 2.0, 2.0, 1.0])
        output_layer = output_dataset.add_layer_like(layer, layer.name)
        transform_affine(layer, output_layer, scale_by_2)
        ```
    """
    forward_transform = AffineTransform(affine_matrix)
    if output_bbox is None:
        if not isinstance(input_layer.bounding_box, BoundingBox):
            raise ValueError(
                "transform_affine is only supported for layers with 3D (x, y, z) "
                + f"bounding boxes, got {type(input_layer.bounding_box).__name__} for layer {input_layer.name}."
            )
        output_bbox = forward_transform.transform_bbox(input_layer.bounding_box)
    return transform(
        input_layer,
        output_layer,
        forward_transform.inverse(),
        mag=mag,
        output_bbox=output_bbox,
        translate_to_positive=translate_to_positive,
        input_mask_layer=input_mask_layer,
        fill_value=fill_value,
        chunk_shape=chunk_shape,
        buffer_shape=buffer_shape,
        executor=executor,
        progress_desc=progress_desc,
    )
