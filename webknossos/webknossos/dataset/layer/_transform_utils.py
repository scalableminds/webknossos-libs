import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from webknossos.geometry import BoundingBox, Mag, NDBoundingBox, Vec3Int

from .view import MagView, View

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
        return  # nothing maps into this tile, the buffer stays zero
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
) -> None:
    (output_view, _i) = args
    try:
        chunk_bbox = output_view.bounding_box
        output_shape = chunk_bbox.in_mag(mag).size
        output_data = np.zeros(
            (input_mag_view.layer.num_channels, *output_shape),
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

        # Always write the full chunk so that the previous content of the output
        # bounding box is overwritten deterministically.
        output_view.write(output_data)
    except Exception as exc:
        logger.exception(
            f"Transforming chunk {output_view.bounding_box} failed with {exc}"
        )
        raise exc
