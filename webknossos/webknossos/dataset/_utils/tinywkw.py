"""Read-only WKW format reader compatible with UPath."""

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


class BlockType(IntEnum):
    RAW = 1
    LZ4 = 2
    LZ4HC = 3


def _morton_encode(block: Vec3Int) -> int:
    """Interleave bits of block coordinates into a Morton (Z-order) index.

    Bit positions: x→0,3,6,…; y→1,4,7,…; z→2,5,8,…
    """
    result = 0
    for bit in range(15):
        result |= ((block.x >> bit) & 1) << (3 * bit)
        result |= ((block.y >> bit) & 1) << (3 * bit + 1)
        result |= ((block.z >> bit) & 1) << (3 * bit + 2)
    return result


@dataclass(frozen=True)
class WkwHeader:
    block_len: int
    file_len: int
    block_type: BlockType
    voxel_dtype: np.dtype
    num_channels: int
    data_offset: int

    @classmethod
    def from_bytes(cls, data: bytes) -> "WkwHeader":
        assert data[0:3] == b"WKW"
        assert data[3] == 1
        per_dim_log2 = data[4]
        block_len = 1 << (per_dim_log2 & 0x0F)
        file_len = 1 << ((per_dim_log2 >> 4) & 0x0F)
        block_type = BlockType(data[5])
        voxel_dtype = VOXEL_TYPES[data[6]]
        voxel_size = data[7]
        data_offset = struct.unpack_from("<Q", data, 8)[0]
        num_channels = voxel_size // voxel_dtype.itemsize
        return cls(
            block_len=block_len,
            file_len=file_len,
            block_type=block_type,
            voxel_dtype=voxel_dtype,
            num_channels=num_channels,
            data_offset=data_offset,
        )


class WkwDataset:
    """Read-only WKW dataset reader using UPath for remote-compatible I/O."""

    def __init__(self, path: UPath, header: WkwHeader) -> None:
        self._path = path
        self._header = header

    @classmethod
    def open(cls, path: UPath) -> "WkwDataset":
        """Open a WKW dataset at `path` (must contain `header.wkw`)."""
        header = WkwHeader.from_bytes((path / "header.wkw").read_bytes())
        return cls(path, header)

    def read(self, bbox: NormalizedBoundingBox) -> np.ndarray:
        """Read voxel data for the given bounding box.

        Returns array of shape (num_channels, x, y, z) in Fortran order.
        Out-of-bounds regions are zero-filled.
        """
        assert bbox.axes == ("c", "x", "y", "z")
        bbox = bbox.denormalize()
        assert isinstance(bbox, BoundingBox)
        header = self._header
        voxels_per_shard = header.file_len * header.block_len

        size = bbox.size
        output = np.zeros(
            (header.num_channels,) + size.to_tuple(),
            dtype=header.voxel_dtype,
            order="F",
        )

        shard_shape = Vec3Int.full(voxels_per_shard)
        for shard_bbox in bbox.chunk(shard_shape, chunk_border_alignments=shard_shape):
            file_index = shard_bbox.topleft // voxels_per_shard
            self._read_file_into(file_index, bbox, output)
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
        voxels_per_shard = header.file_len * header.block_len
        block_bytes = (
            header.block_len**3 * header.num_channels * header.voxel_dtype.itemsize
        )

        file_data = file_path.read_bytes()
        # Each shard file stores its own data_offset; header.wkw may have 0.
        shard_data_offset = struct.unpack_from("<Q", file_data, 8)[0]

        jump_table = None
        if header.block_type in (BlockType.LZ4, BlockType.LZ4HC):
            import lz4.block  # noqa: PLC0415

            num_blocks = header.file_len**3
            jump_table = np.frombuffer(file_data[16 : 16 + num_blocks * 8], dtype="<u8")

        file_origin = file_index * voxels_per_shard
        shard_bbox = bbox.intersected_with(
            bbox.with_topleft(file_origin).with_size(Vec3Int.full(voxels_per_shard))
        )

        block_shape = Vec3Int.full(header.block_len)
        for block_bbox in shard_bbox.chunk(
            block_shape, chunk_border_alignments=block_shape
        ):
            local_block = (
                block_bbox.topleft // header.block_len - file_index * header.file_len
            )
            morton_index = _morton_encode(local_block)
            if header.block_type == BlockType.RAW:
                byte_offset = shard_data_offset + morton_index * block_bytes
                raw_block = file_data[byte_offset : byte_offset + block_bytes]
            else:
                compressed_start = (
                    int(jump_table[morton_index - 1])  # type: ignore[index]
                    if morton_index > 0
                    else shard_data_offset
                )
                compressed_end = int(jump_table[morton_index])  # type: ignore[index]
                raw_block = lz4.block.decompress(
                    file_data[compressed_start:compressed_end],
                    uncompressed_size=block_bytes,
                )
            block_data = np.frombuffer(raw_block, dtype=header.voxel_dtype).reshape(
                (
                    header.num_channels,
                    header.block_len,
                    header.block_len,
                    header.block_len,
                ),
                order="F",
            )

            all_sel = (slice(None),)
            block_sel = all_sel + block_bbox.offset(-block_bbox.topleft).to_slices()
            output_sel = all_sel + block_bbox.offset(-bbox.topleft).to_slices()
            output[output_sel] = block_data[block_sel]
