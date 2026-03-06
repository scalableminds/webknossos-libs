"""Read-only WKW format reader compatible with UPath."""

import io
import struct
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from upath import UPath

from webknossos.geometry import BoundingBox, NormalizedBoundingBox, Vec3Int

VOXEL_TYPES: dict[int, np.dtype] = {
    0x01: np.dtype("uint8"),
    0x02: np.dtype("uint16"),
    0x03: np.dtype("uint32"),
    0x04: np.dtype("uint64"),
    0x05: np.dtype("float32"),
    0x06: np.dtype("float64"),
}


class ChunkType(IntEnum):
    RAW = 1
    LZ4 = 2
    LZ4HC = 3


def _morton_encode(chunk: Vec3Int) -> int:
    """Interleave bits of chunk coordinates into a Morton (Z-order) index.

    Bit positions: x→0,3,6,…; y→1,4,7,…; z→2,5,8,…
    """
    result = 0
    for bit in range(15):
        result |= ((chunk.x >> bit) & 1) << (3 * bit)
        result |= ((chunk.y >> bit) & 1) << (3 * bit + 1)
        result |= ((chunk.z >> bit) & 1) << (3 * bit + 2)
    return result


@dataclass(frozen=True)
class WkwHeader:
    chunk_shape: Vec3Int
    shard_shape: Vec3Int
    chunk_type: ChunkType
    voxel_dtype: np.dtype
    num_channels: int
    data_offset: int

    @classmethod
    def from_bytes(cls, data: bytes) -> "WkwHeader":
        assert data[0:3] == b"WKW"
        assert data[3] == 1
        per_dim_log2 = data[4]
        chunk_shape = Vec3Int.full(1 << (per_dim_log2 & 0x0F))
        shard_shape = Vec3Int.full(1 << ((per_dim_log2 >> 4) & 0x0F))
        chunk_type = ChunkType(data[5])
        voxel_dtype = VOXEL_TYPES[data[6]]
        voxel_size = data[7]
        data_offset = struct.unpack_from("<Q", data, 8)[0]
        num_channels = voxel_size // voxel_dtype.itemsize
        return cls(
            chunk_shape=chunk_shape,
            shard_shape=shard_shape,
            chunk_type=chunk_type,
            voxel_dtype=voxel_dtype,
            num_channels=num_channels,
            data_offset=data_offset,
        )

    @property
    def block_len(self) -> int:
        return self.chunk_shape.x

    @property
    def file_len(self) -> int:
        return self.shard_shape.x

    @property
    def block_type(self) -> int:
        return self.chunk_type.value

    @property
    def voxel_type(self) -> np.dtype:
        return self.voxel_type


class WkwDataset:
    """Read-only WKW dataset reader using UPath for remote-compatible I/O."""

    def __init__(self, path: UPath, header: WkwHeader) -> None:
        self._path = path
        self._header = header

    @property
    def header(self) -> WkwHeader:
        return self._header

    @classmethod
    def open(cls, path: UPath) -> "WkwDataset":
        """Open a WKW dataset at `path` (must contain `header.wkw`)."""
        header = WkwHeader.from_bytes((path / "header.wkw").read_bytes())
        return cls(path, header)

    def close(self) -> None:
        """Close the dataset."""
        pass

    def read(self, topleft: Vec3Int, size: Vec3Int) -> np.ndarray:
        """Read voxel data for the given bounding box.

        Returns array of shape (num_channels, x, y, z) in Fortran order.
        Out-of-bounds regions are zero-filled.
        """
        return self.read_bbox(
            BoundingBox(topleft, size).normalize_axes(self.header.num_channels)
        )

    def read_bbox(self, bbox: NormalizedBoundingBox) -> np.ndarray:
        """Read voxel data for the given bounding box.

        Returns array of shape (num_channels, x, y, z) in Fortran order.
        Out-of-bounds regions are zero-filled.
        """
        assert bbox.axes == ("c", "x", "y", "z")
        bbox_xyz = bbox.denormalize()
        assert isinstance(bbox_xyz, BoundingBox)
        header = self._header
        voxels_per_shard = header.shard_shape * header.chunk_shape

        size = bbox_xyz.size
        output = np.zeros(
            (header.num_channels,) + size.to_tuple(),
            dtype=header.voxel_dtype,
            order="F",
        )

        for shard_bbox in bbox_xyz.chunk(
            voxels_per_shard, chunk_border_alignments=voxels_per_shard
        ):
            file_index = shard_bbox.topleft // voxels_per_shard
            self._read_file_into(file_index, bbox_xyz, output)
        return output

    def _read_file_into(
        self,
        file_index: Vec3Int,
        bbox: BoundingBox,
        output: np.ndarray,
    ) -> None:
        file_path = (
            self._path
            / f"z{file_index.z}"
            / f"y{file_index.y}"
            / f"x{file_index.x}.wkw"
        )
        if not file_path.exists():
            return
        header = self._header
        voxels_per_shard = header.shard_shape * header.chunk_shape
        chunk_bytes = (
            header.chunk_shape.x
            * header.chunk_shape.y
            * header.chunk_shape.z
            * header.num_channels
            * header.voxel_dtype.itemsize
        )

        file_origin = file_index * voxels_per_shard
        shard_bbox = bbox.intersected_with(
            bbox.with_topleft(file_origin).with_size(voxels_per_shard)
        )

        # if full shard is requested, pre-load the whole file
        full_shard = shard_bbox.size == voxels_per_shard
        f_ctx = (
            io.BytesIO(file_path.read_bytes()) if full_shard else file_path.open("rb")
        )

        with f_ctx as f:
            # Each shard file stores its own data_offset; header.wkw may have 0.
            f.seek(8)
            shard_data_offset = struct.unpack("<Q", f.read(8))[0]

            jump_table = None
            if header.chunk_type in (ChunkType.LZ4, ChunkType.LZ4HC):
                import lz4.block  # noqa: PLC0415

                num_chunks = (
                    header.shard_shape.x * header.shard_shape.y * header.shard_shape.z
                )
                f.seek(16)
                jump_table = np.frombuffer(f.read(num_chunks * 8), dtype="<u8")

            for chunk_bbox in shard_bbox.chunk(
                header.chunk_shape, chunk_border_alignments=header.chunk_shape
            ):
                local_chunk = (
                    chunk_bbox.topleft // header.chunk_shape
                    - file_index * header.shard_shape
                )
                morton_index = _morton_encode(local_chunk)
                if header.chunk_type == ChunkType.RAW:
                    f.seek(shard_data_offset + morton_index * chunk_bytes)
                    raw_chunk = f.read(chunk_bytes)
                else:
                    compressed_start = (
                        int(jump_table[morton_index - 1])  # type: ignore[index]
                        if morton_index > 0
                        else shard_data_offset
                    )
                    compressed_end = int(jump_table[morton_index])  # type: ignore[index]
                    f.seek(compressed_start)
                    raw_chunk = lz4.block.decompress(
                        f.read(compressed_end - compressed_start),
                        uncompressed_size=chunk_bytes,
                    )
                chunk_data = np.frombuffer(raw_chunk, dtype=header.voxel_dtype).reshape(
                    (header.num_channels,) + header.chunk_shape.to_tuple(),
                    order="F",
                )

                all_sel = (slice(None),)
                chunk_sel = all_sel + chunk_bbox.offset(-chunk_bbox.topleft).to_slices()
                output_sel = all_sel + chunk_bbox.offset(-bbox.topleft).to_slices()
                output[output_sel] = chunk_data[chunk_sel]
