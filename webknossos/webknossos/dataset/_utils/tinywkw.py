"""WKW format reader/writer compatible with UPath."""

import io
import struct
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import IntEnum

import lz4.block
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
VOXEL_TYPE_KEYS: dict[np.dtype, int] = {v: k for k, v in VOXEL_TYPES.items()}

ALL_SEL = (slice(None),)


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
    voxel_type: np.dtype
    num_channels: int
    data_offset: int | None = None

    @classmethod
    def from_bytes(cls, data: bytes) -> "WkwHeader":
        assert data[0:3] == b"WKW"
        assert data[3] == 1
        per_dim_log2 = data[4]
        chunk_shape = Vec3Int.full(1 << (per_dim_log2 & 0x0F))
        shard_shape = Vec3Int.full(1 << ((per_dim_log2 >> 4) & 0x0F)) * chunk_shape
        chunk_type = ChunkType(data[5])
        voxel_type = VOXEL_TYPES[data[6]]
        voxel_size = data[7]
        data_offset = struct.unpack_from("<Q", data, 8)[0]
        num_channels = voxel_size // voxel_type.itemsize
        return cls(
            chunk_shape=chunk_shape,
            shard_shape=shard_shape,
            chunk_type=chunk_type,
            voxel_type=voxel_type,
            num_channels=num_channels,
            data_offset=data_offset,
        )

    def to_bytes(self, data_offset: int | None = None) -> bytes:
        chunk_len_log2 = self.block_len.bit_length() - 1
        shard_len_log2 = self.file_len.bit_length() - 1
        per_dim_log2 = (chunk_len_log2 & 0x0F) | ((shard_len_log2 & 0x0F) << 4)
        voxel_size = self.num_channels * self.voxel_type.itemsize
        if data_offset is None:
            num_chunks = self.file_len**3
            data_offset = (
                16 if self.chunk_type == ChunkType.RAW else 16 + num_chunks * 8
            )
        return (
            b"WKW\x01"
            + bytes(
                [
                    per_dim_log2,
                    self.chunk_type.value,
                    VOXEL_TYPE_KEYS[self.voxel_type],
                    voxel_size,
                ]
            )
            + struct.pack("<Q", data_offset)
        )

    # ------------------------------------------------------------------
    # Compatibility with wkw.Header
    # ------------------------------------------------------------------

    @property
    def block_len(self) -> int:
        return self.chunk_shape.x

    @property
    def file_len(self) -> int:
        return self.shard_shape.x // self.chunk_shape.x

    @property
    def block_type(self) -> int:
        return self.chunk_type.value


class WkwDataset:
    """WKW dataset reader/writer using UPath for remote-compatible I/O."""

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

    @classmethod
    def create(cls, path: UPath, header: WkwHeader) -> "WkwDataset":
        """Create a new WKW dataset at `path`, writing `header.wkw` with data_offset=0."""
        path.mkdir(parents=True, exist_ok=True)
        (path / "header.wkw").write_bytes(header.to_bytes(data_offset=0))
        return cls(path, header)

    def close(self) -> None:
        """Close the dataset."""
        pass

    def _shard_path(self, shard_address: Vec3Int) -> UPath:
        return (
            self._path
            / f"z{shard_address.z}"
            / f"y{shard_address.y}"
            / f"x{shard_address.x}.wkw"
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

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

        size = bbox_xyz.size
        output = np.zeros(
            (header.num_channels,) + size.to_tuple(),
            dtype=header.voxel_type,
            order="F",
        )

        for shard_bbox in bbox_xyz.chunk(
            header.shard_shape, chunk_border_alignments=header.shard_shape
        ):
            shard_address = shard_bbox.topleft // header.shard_shape
            self._read_shard_into(shard_address, bbox_xyz, output)
        return output

    def _read_shard_into(
        self,
        shard_address: Vec3Int,
        bbox: BoundingBox,
        output: np.ndarray,
    ) -> None:
        file_path = self._shard_path(shard_address)
        if not file_path.exists():
            return
        header = self._header
        num_chunk_bytes = (
            header.chunk_shape.x
            * header.chunk_shape.y
            * header.chunk_shape.z
            * header.num_channels
            * header.voxel_type.itemsize
        )

        shard_topleft = shard_address * header.shard_shape
        full_shard_bbox = BoundingBox(shard_topleft, header.shard_shape)
        shard_bbox = bbox.intersected_with(full_shard_bbox)

        # if full shard is requested, pre-load the whole file
        f_ctx = (
            io.BytesIO(file_path.read_bytes())
            if shard_bbox == full_shard_bbox
            else file_path.open("rb")
        )

        with f_ctx as f:
            # Each shard file stores its own data_offset; header.wkw may have 0.
            f.seek(8)
            shard_data_offset = struct.unpack("<Q", f.read(8))[0]

            jump_table = None
            if header.chunk_type in (ChunkType.LZ4, ChunkType.LZ4HC):
                num_chunks = header.file_len**3
                f.seek(16)
                jump_table = np.frombuffer(f.read(num_chunks * 8), dtype="<u8")

            chunk_bboxes = sorted(
                shard_bbox.chunk(
                    header.chunk_shape, chunk_border_alignments=header.chunk_shape
                ),
                key=lambda bb: _morton_encode(
                    bb.offset(-shard_topleft).topleft // header.chunk_shape
                ),
            )

            if header.chunk_type == ChunkType.RAW:
                raw_chunks = []
                for chunk_bbox in chunk_bboxes:
                    morton_index = _morton_encode(
                        chunk_bbox.offset(-shard_topleft).topleft // header.chunk_shape
                    )
                    f.seek(shard_data_offset + morton_index * num_chunk_bytes)
                    raw_chunks.append(f.read(num_chunk_bytes))
            else:
                assert jump_table is not None
                # Phase 1: gather compressed bytes (sequential I/O)
                compressed_bufs = []
                for chunk_bbox in chunk_bboxes:
                    morton_index = _morton_encode(
                        chunk_bbox.offset(-shard_topleft).topleft // header.chunk_shape
                    )
                    compressed_start = (
                        int(jump_table[morton_index - 1])
                        if morton_index > 0
                        else shard_data_offset
                    )
                    compressed_end = int(jump_table[morton_index])
                    f.seek(compressed_start)
                    compressed_bufs.append(f.read(compressed_end - compressed_start))
                # Phase 2: decompress in parallel
                with ThreadPoolExecutor() as executor:
                    raw_chunks = list(
                        executor.map(
                            lambda cb: lz4.block.decompress(
                                cb, uncompressed_size=num_chunk_bytes
                            ),
                            compressed_bufs,
                        )
                    )

        for chunk_bbox, raw_chunk in zip(chunk_bboxes, raw_chunks):
            chunk_data = np.frombuffer(raw_chunk, dtype=header.voxel_type).reshape(
                (header.num_channels,) + header.chunk_shape.to_tuple(),
                order="F",
            )
            # chunk_origin is the aligned start of this chunk in world coordinates
            chunk_origin = (
                chunk_bbox.topleft // header.chunk_shape
            ) * header.chunk_shape
            chunk_sel = ALL_SEL + chunk_bbox.offset(-chunk_origin).to_slices()
            output_sel = ALL_SEL + chunk_bbox.offset(-bbox.topleft).to_slices()
            output[output_sel] = chunk_data[chunk_sel]

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(self, topleft: Vec3Int, data: np.ndarray) -> None:
        """Write voxel data starting at `topleft`.

        `data` must have shape (num_channels, x, y, z) in Fortran order.
        """
        size = Vec3Int(*data.shape[1:])
        self.write_bbox(
            BoundingBox(topleft, size).normalize_axes(self.header.num_channels),
            data,
        )

    def write_bbox(self, bbox: NormalizedBoundingBox, data: np.ndarray) -> None:
        """Write voxel data for the given bounding box.

        `data` must have shape (num_channels, x, y, z) in Fortran order.
        """
        assert bbox.axes == ("c", "x", "y", "z")
        bbox_xyz = bbox.denormalize()
        assert isinstance(bbox_xyz, BoundingBox)
        header = self._header

        for shard_bbox in bbox_xyz.chunk(
            header.shard_shape, chunk_border_alignments=header.shard_shape
        ):
            shard_address = shard_bbox.topleft // header.shard_shape
            self._write_shard_from(shard_address, bbox_xyz, data)

    def _write_shard_from(
        self,
        shard_address: Vec3Int,
        bbox: BoundingBox,
        data: np.ndarray,
    ) -> None:
        header = self._header
        shard_topleft = shard_address * header.shard_shape
        full_shard_bbox = BoundingBox(shard_topleft, header.shard_shape)
        shard_bbox = bbox.intersected_with(full_shard_bbox)

        if shard_bbox == full_shard_bbox:
            # Fast path: the entire shard is being written
            data_sel = ALL_SEL + shard_bbox.offset(-bbox.topleft).to_slices()
            shard_buffer = data[data_sel]
            self._write_shard(shard_address, shard_buffer)
            return

        # Allocate shard buffer; read existing file if present (preserves unwritten regions)
        shard_buffer = np.zeros(
            (header.num_channels,) + header.shard_shape.to_tuple(),
            dtype=header.voxel_type,
            order="F",
        )
        if self._shard_path(shard_address).exists():
            self._read_shard_into(shard_address, full_shard_bbox, shard_buffer)

        # Merge new data into shard buffer
        shard_sel = ALL_SEL + shard_bbox.offset(-shard_topleft).to_slices()
        data_sel = ALL_SEL + shard_bbox.offset(-bbox.topleft).to_slices()
        shard_buffer[shard_sel] = data[data_sel]

        self._write_shard(shard_address, shard_buffer)

    def _write_shard(self, shard_address: Vec3Int, shard_buffer: np.ndarray) -> None:
        file_path = self._shard_path(shard_address)
        if not np.any(shard_buffer):
            if file_path.exists():
                file_path.unlink()
            return

        header = self._header
        shard_topleft = shard_address * header.shard_shape
        full_shard_bbox = BoundingBox(shard_topleft, header.shard_shape)

        num_chunks = header.file_len**3

        # Build chunk payloads in Morton order
        raw_chunks: list[bytes] = [b""] * num_chunks
        for chunk_bbox in full_shard_bbox.chunk(
            header.chunk_shape, chunk_border_alignments=header.chunk_shape
        ):
            morton_index = _morton_encode(
                chunk_bbox.offset(-full_shard_bbox.topleft).topleft
                // header.chunk_shape
            )
            chunk_sel = ALL_SEL + chunk_bbox.offset(-shard_topleft).to_slices()
            raw_chunks[morton_index] = shard_buffer[chunk_sel].tobytes(order="F")

        file_path.parent.mkdir(parents=True, exist_ok=True)

        if header.chunk_type == ChunkType.RAW:
            with file_path.open("wb") as f:
                f.write(header.to_bytes())
                for raw_chunk in raw_chunks:
                    f.write(raw_chunk)
        else:
            compression_mode = (
                "high_compression"
                if header.chunk_type == ChunkType.LZ4HC
                else "default"
            )
            with ThreadPoolExecutor() as executor:
                compressed_chunks = list(
                    executor.map(
                        lambda rc: lz4.block.compress(
                            rc, store_size=False, mode=compression_mode
                        ),
                        raw_chunks,
                    )
                )
            data_offset = 16 + num_chunks * 8
            jump_table = np.zeros(num_chunks, dtype="<u8")
            offset = data_offset
            for i, c in enumerate(compressed_chunks):
                offset += len(c)
                jump_table[i] = offset
            with file_path.open("wb") as f:
                f.write(header.to_bytes())
                f.write(jump_table.tobytes())
                for c in compressed_chunks:
                    f.write(c)
