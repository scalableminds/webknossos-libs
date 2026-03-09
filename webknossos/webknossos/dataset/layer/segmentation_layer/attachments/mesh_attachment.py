import json
import math
import struct
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from os import PathLike
from typing import Literal

import numpy as np
from upath import UPath

from webknossos.dataset_properties import AttachmentDataFormat

from ._utils import read_zarr3_array, write_zarr3_array
from .attachment import Attachment


@dataclass
class MeshFragment:
    """One Draco-encoded mesh fragment at a grid position."""

    position: tuple[int, int, int]  # grid position (x, y, z) as uint32
    data: bytes  # Draco-encoded triangle mesh bytes


@dataclass
class MeshLod:
    """One level of detail for a segment."""

    scale: float  # lod_scales[i]
    vertex_offset: tuple[float, float, float]  # vertex_offsets[i]
    fragments: list[MeshFragment]


@dataclass
class SegmentMesh:
    """All LoDs for one segment."""

    segment_id: int
    chunk_shape: tuple[float, float, float]  # float32 voxel size of finest chunk
    grid_origin: tuple[float, float, float]  # float32 origin of chunk grid
    lods: list[MeshLod]


@dataclass
class MeshfileMetadata:
    """Mesh-level metadata stored in the meshfile zarr.json."""

    global_bounding_box: tuple[int, int, int, int, int, int]  # x,y,z,w,h,d voxels
    mag: tuple[int, int, int]
    mapping_name: str = ""
    vertex_quantization_bits_per_lod: list[int] = field(default_factory=list)
    transform: np.ndarray = field(
        default_factory=lambda: np.eye(3, 4, dtype=np.float32)
    )
    lod_scale_multiplier: float = 1.0
    unit: str = "nm"


def _pack_manifest(segment: SegmentMesh) -> bytes:
    """Pack the manifest binary for a segment (all little-endian)."""
    buf = bytearray()
    # chunk_shape float32 (3,)
    buf += struct.pack("<3f", *segment.chunk_shape)
    # grid_origin float32 (3,)
    buf += struct.pack("<3f", *segment.grid_origin)
    # num_lods uint32
    num_lods = len(segment.lods)
    buf += struct.pack("<I", num_lods)
    # lod_scales float32 (num_lods,)
    buf += struct.pack(f"<{num_lods}f", *[lod.scale for lod in segment.lods])
    # vertex_offsets float32 (num_lods, 3)  row-major
    for lod in segment.lods:
        buf += struct.pack("<3f", *lod.vertex_offset)
    # num_fragments_per_lod uint32 (num_lods,)
    buf += struct.pack(f"<{num_lods}I", *[len(lod.fragments) for lod in segment.lods])
    # For each LoD: fragment_positions column-major, fragment_byte_sizes
    for lod in segment.lods:
        n_frags = len(lod.fragments)
        if n_frags > 0:
            # column-major: all X, then all Y, then all Z
            xs = [f.position[0] for f in lod.fragments]
            ys = [f.position[1] for f in lod.fragments]
            zs = [f.position[2] for f in lod.fragments]
            buf += struct.pack(f"<{n_frags}I", *xs)
            buf += struct.pack(f"<{n_frags}I", *ys)
            buf += struct.pack(f"<{n_frags}I", *zs)
            buf += struct.pack(f"<{n_frags}I", *[len(f.data) for f in lod.fragments])
    return bytes(buf)


def _parse_manifest(
    data: bytes, segment_id: int
) -> tuple[SegmentMesh, list[list[int]]]:
    """Parse a manifest binary. Returns (SegmentMesh, lods_byte_sizes).

    The SegmentMesh fragments have empty `data` bytes; callers must fill them
    from the neuroglancer array using the returned lods_byte_sizes.
    """
    offset = 0

    def read_fmt(fmt: str) -> tuple[int | float, ...]:
        nonlocal offset
        size = struct.calcsize(fmt)
        values = struct.unpack_from(fmt, data, offset)
        offset += size
        return values  # type: ignore[return-value]

    chunk_shape = read_fmt("<3f")
    grid_origin = read_fmt("<3f")
    (num_lods_raw,) = read_fmt("<I")
    num_lods = int(num_lods_raw)
    lod_scales = read_fmt(f"<{num_lods}f")
    # vertex_offsets: (num_lods, 3) row-major
    vertex_offsets: list[tuple[int | float, ...]] = [
        read_fmt("<3f") for _ in range(num_lods)
    ]
    num_frags_per_lod = read_fmt(f"<{num_lods}I")

    lods_frags_positions: list[list[tuple[int, int, int]]] = []
    lods_byte_sizes: list[list[int]] = []
    for lod_idx in range(num_lods):
        n_frags = int(num_frags_per_lod[lod_idx])
        if n_frags > 0:
            xs = read_fmt(f"<{n_frags}I")
            ys = read_fmt(f"<{n_frags}I")
            zs = read_fmt(f"<{n_frags}I")
            byte_sizes = read_fmt(f"<{n_frags}I")
            positions = [(int(xs[i]), int(ys[i]), int(zs[i])) for i in range(n_frags)]
            sizes = [int(s) for s in byte_sizes]
        else:
            positions = []
            sizes = []
        lods_frags_positions.append(positions)
        lods_byte_sizes.append(sizes)

    lods: list[MeshLod] = []
    for lod_idx in range(num_lods):
        lod = MeshLod(
            scale=float(lod_scales[lod_idx]),
            vertex_offset=(
                float(vertex_offsets[lod_idx][0]),
                float(vertex_offsets[lod_idx][1]),
                float(vertex_offsets[lod_idx][2]),
            ),
            fragments=[
                MeshFragment(position=pos, data=b"")
                for pos in lods_frags_positions[lod_idx]
            ],
        )
        lods.append(lod)

    segment_mesh = SegmentMesh(
        segment_id=segment_id,
        chunk_shape=(
            float(chunk_shape[0]),
            float(chunk_shape[1]),
            float(chunk_shape[2]),
        ),
        grid_origin=(
            float(grid_origin[0]),
            float(grid_origin[1]),
            float(grid_origin[2]),
        ),
        lods=lods,
    )
    return segment_mesh, lods_byte_sizes


def _murmurhash3_bucket(segment_id: int, n_buckets: int) -> int:
    import mmh3

    return mmh3.hash128(struct.pack("<Q", segment_id), seed=0, x64arch=True) % n_buckets


class MeshAttachment(Attachment):
    data_format: Literal[AttachmentDataFormat.Zarr3, AttachmentDataFormat.HDF5]
    container_name = "meshes"
    type_name = "mesh"

    @classmethod
    def create(
        cls,
        path: str | PathLike | UPath,
        segments: Sequence[SegmentMesh],
        metadata: MeshfileMetadata,
    ) -> "MeshAttachment":
        """Create and write a Zarr v3 mesh attachment.

        `path` is the mesh group directory (e.g. `…/mesh_view`).
        The directory name is used as the attachment name.

        Returns a MeshAttachment usable directly with
        `seg_layer.attachments.add_attachment_as_copy(attachment)`.
        """
        path = UPath(path)
        path.mkdir(parents=True, exist_ok=True)

        n_segments = len(segments)
        if n_segments == 0:
            raise ValueError("segments must not be empty.")

        n_buckets = math.ceil(n_segments / 0.75)
        n_lod = len(segments[0].lods) if segments else 0

        # Compute bucket indices and sort for determinism
        keyed = [
            (_murmurhash3_bucket(seg.segment_id, n_buckets), seg.segment_id, seg)
            for seg in segments
        ]
        keyed.sort(key=lambda t: (t[0], t[1]))

        # Pack neuroglancer buffer and build buckets array
        neuroglancer_buf = bytearray()
        bucket_rows: list[
            tuple[int, int, int]
        ] = []  # (segment_id, manifest_start, manifest_end)

        for bucket_idx, segment_id, seg in keyed:
            # Concatenate all Draco bytes: LoD 0 frags, LoD 1 frags, ...
            draco_bytes = bytearray()
            for lod in seg.lods:
                for frag in lod.fragments:
                    draco_bytes += frag.data

            manifest_bytes = _pack_manifest(seg)
            manifest_start = len(neuroglancer_buf) + len(draco_bytes)
            manifest_end = manifest_start + len(manifest_bytes)

            neuroglancer_buf += draco_bytes
            neuroglancer_buf += manifest_bytes

            bucket_rows.append((segment_id, manifest_start, manifest_end))

        # Build buckets array (n_segments, 3) uint64
        buckets_array = np.array(bucket_rows, dtype=np.uint64).reshape(n_segments, 3)

        # Build bucket_offsets CSR (n_buckets+1,) uint64
        # First, compute which bucket each row belongs to
        bucket_assignments = [
            _murmurhash3_bucket(seg_id, n_buckets) for _, seg_id, _ in keyed
        ]
        bucket_offsets = np.zeros(n_buckets + 1, dtype=np.uint64)
        # Count entries per bucket
        counts = np.zeros(n_buckets, dtype=np.int64)
        for b in bucket_assignments:
            counts[b] += 1
        # Build CSR offsets
        for i in range(n_buckets):
            bucket_offsets[i + 1] = bucket_offsets[i] + counts[i]

        neuroglancer_array = np.frombuffer(bytes(neuroglancer_buf), dtype=np.uint8)

        # Write root zarr.json
        root_meta = {
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {
                "voxelytics": {
                    "artifact_schema_version": 9,
                    "artifact_class": "MeshfileArtifact",
                }
            },
        }
        (path / "zarr.json").write_text(json.dumps(root_meta, indent=2))

        # Write meshfile/zarr.json
        meshfile_path = path / "meshfile"
        meshfile_path.mkdir(parents=True, exist_ok=True)

        transform_list = metadata.transform.tolist()
        meshfile_meta = {
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {
                "voxelytics": {
                    "artifact_schema_version": 9,
                    "artifact_class": "MeshfileArtifact",
                    "version": 9,
                    "global_bounding_box": list(metadata.global_bounding_box),
                    "mag": list(metadata.mag),
                    "mapping_name": metadata.mapping_name,
                    "hash_function": "murmurhash3_x64_128",
                    "n_buckets": n_buckets,
                    "mesh_format": "draco",
                    "n_lod": n_lod,
                    "vertex_quantization_bits_per_lod": metadata.vertex_quantization_bits_per_lod,
                    "transform": transform_list,
                    "lod_scale_multiplier": float(metadata.lod_scale_multiplier),
                    "unit": metadata.unit,
                }
            },
        }
        (meshfile_path / "zarr.json").write_text(json.dumps(meshfile_meta, indent=2))

        # Write arrays
        BUCKET_OFFSETS_CHUNK = 64 * 1024  # 64 KB
        BUCKET_OFFSETS_SHARD = 128 * 1024 * 1024  # 128 MB
        BUCKETS_CHUNK = 128 * 1024  # 128 KB
        BUCKETS_SHARD = 256 * 1024 * 1024  # 256 MB
        NEURO_CHUNK = 64 * 1024  # 64 KB
        NEURO_SHARD = 512 * 1024 * 1024  # 512 MB

        write_zarr3_array(
            meshfile_path / "bucket_offsets",
            bucket_offsets,
            dtype="uint64",
            target_chunk_size_bytes=BUCKET_OFFSETS_CHUNK,
            target_shard_size_bytes=BUCKET_OFFSETS_SHARD,
        )
        write_zarr3_array(
            meshfile_path / "buckets",
            buckets_array,
            dtype="uint64",
            target_chunk_size_bytes=BUCKETS_CHUNK,
            target_shard_size_bytes=BUCKETS_SHARD,
        )
        write_zarr3_array(
            meshfile_path / "neuroglancer",
            neuroglancer_array,
            dtype="uint8",
            target_chunk_size_bytes=NEURO_CHUNK,
            target_shard_size_bytes=NEURO_SHARD,
        )

        return cls.from_path_and_name(
            path,
            path.name,
            data_format=AttachmentDataFormat.Zarr3,
        )

    def _read_meshfile_attrs(self) -> dict:  # type: ignore[type-ignore]
        import json

        meta_path = self.path / "meshfile" / "zarr.json"
        return json.loads(meta_path.read_text())["attributes"]["voxelytics"]  # type: ignore[no-any-return]

    def parse_segment(self, segment_id: int) -> SegmentMesh:
        """Read and parse the mesh for a single segment by ID."""
        if self.data_format != AttachmentDataFormat.Zarr3:
            raise NotImplementedError(
                f"parse_segment() only supports Zarr3 format, got {self.data_format}"
            )

        attrs = self._read_meshfile_attrs()
        n_buckets = int(attrs["n_buckets"])
        meshfile_path = self.path / "meshfile"

        bucket_idx = _murmurhash3_bucket(segment_id, n_buckets)

        bucket_offsets = read_zarr3_array(meshfile_path / "bucket_offsets")
        start = int(bucket_offsets[bucket_idx])
        end = int(bucket_offsets[bucket_idx + 1])

        if start == end:
            raise KeyError(f"Segment {segment_id} not found in mesh file.")

        buckets = read_zarr3_array(meshfile_path / "buckets")
        bucket_slice = buckets[start:end]

        row_idx = None
        for i in range(len(bucket_slice)):
            if int(bucket_slice[i, 0]) == segment_id:
                row_idx = i
                break

        if row_idx is None:
            raise KeyError(f"Segment {segment_id} not found in mesh file.")

        manifest_start = int(bucket_slice[row_idx, 1])
        manifest_end = int(bucket_slice[row_idx, 2])

        neuroglancer = read_zarr3_array(meshfile_path / "neuroglancer")
        manifest_bytes = bytes(neuroglancer[manifest_start:manifest_end])

        segment_mesh, lods_byte_sizes = _parse_manifest(manifest_bytes, segment_id)

        # Compute draco_start and read Draco data for each LoD
        total_draco_bytes = sum(sum(sizes) for sizes in lods_byte_sizes)
        draco_start = manifest_start - total_draco_bytes

        offset = draco_start
        for lod_idx, lod in enumerate(segment_mesh.lods):
            for frag_idx, frag in enumerate(lod.fragments):
                size = lods_byte_sizes[lod_idx][frag_idx]
                frag.data = bytes(neuroglancer[offset : offset + size])
                offset += size

        return segment_mesh

    def parse_all(self) -> Iterator[SegmentMesh]:
        """Yield SegmentMesh for every segment in the mesh file."""
        if self.data_format != AttachmentDataFormat.Zarr3:
            raise NotImplementedError(
                f"parse_all() only supports Zarr3 format, got {self.data_format}"
            )

        meshfile_path = self.path / "meshfile"

        buckets = read_zarr3_array(meshfile_path / "buckets")
        neuroglancer = read_zarr3_array(meshfile_path / "neuroglancer")

        for row_idx in range(len(buckets)):
            segment_id = int(buckets[row_idx, 0])
            manifest_start = int(buckets[row_idx, 1])
            manifest_end = int(buckets[row_idx, 2])

            manifest_bytes = bytes(neuroglancer[manifest_start:manifest_end])
            segment_mesh, lods_byte_sizes = _parse_manifest(manifest_bytes, segment_id)

            total_draco_bytes = sum(sum(sizes) for sizes in lods_byte_sizes)
            draco_start = manifest_start - total_draco_bytes

            offset = draco_start
            for lod_idx, lod in enumerate(segment_mesh.lods):
                for frag_idx, frag in enumerate(lod.fragments):
                    size = lods_byte_sizes[lod_idx][frag_idx]
                    frag.data = bytes(neuroglancer[offset : offset + size])
                    offset += size

            yield segment_mesh
