from __future__ import annotations

import json
from os import PathLike
from typing import TYPE_CHECKING, Literal

import networkx as nx
import numpy as np
from upath import UPath

from webknossos.dataset._utils.tensorstore_helpers import (
    read_zarr3_array,
    write_zarr3_array,
)
from webknossos.dataset_properties import AttachmentDataFormat
from webknossos.geometry import Vec3Int

from .attachment import Attachment

if TYPE_CHECKING:
    from webknossos.dataset import SegmentationLayer
    from webknossos.proofreading.agglomerate_graph_data import AgglomerateGraphData


def _validate_segment_ids(graph: AgglomerateGraph) -> tuple[int, list[int]]:
    segment_ids: list[int] = list(graph.nodes)
    for sid in segment_ids:
        if not isinstance(sid, int) or sid < 1:
            raise ValueError(
                f"All node labels must be positive integers (1-based), got {sid!r}"
            )

    n_segments = max(segment_ids) if len(segment_ids) > 0 else 0
    if set(segment_ids) != set(range(1, n_segments + 1)):
        missing = sorted(set(range(1, n_segments + 1)) - set(segment_ids))
        raise ValueError(
            f"Segment IDs must be dense (1..{n_segments}) with no gaps, "
            f"missing: {missing}"
        )
    return n_segments, segment_ids


class AgglomerateGraph(nx.Graph):
    """Typed networkx Graph for agglomerate data.
    Node labels are segment IDs (integers, 1-based).
    """

    def add_segment(self, segment_id: int, *, position: Vec3Int) -> None:
        """Add a segment node with a (x, y, z) position."""
        self.add_node(segment_id, position=position)

    def add_affinity_edge(self, u: int, v: int, *, affinity: float) -> None:
        """Add an undirected edge between two segment IDs with an affinity score."""
        self.add_edge(u, v, affinity=affinity)

    def to_agglomerate_graph_data(
        self, segmentation_dtype: np.typing.DTypeLike = np.dtype("uint32")
    ) -> AgglomerateGraphData:
        """Convert to an AgglomerateGraphData (flat numpy-array representation).

        Returns an AgglomerateGraphData with:
        - segments: segment IDs, segmentation_dtype (uint32 or uint64)
        - positions: [x, y, z] per segment, dtype int64
        - edges: [source, target] segment ID pairs, segmentation_dtype (uint32 or uint64)
        - affinities: affinity per edge, dtype float32
        """
        from webknossos.proofreading.agglomerate_graph_data import AgglomerateGraphData

        segmentation_dtype = np.dtype(segmentation_dtype)
        node_list = list(self.nodes)
        segments = np.array(node_list, dtype=segmentation_dtype)
        positions = np.array(
            [list(self.nodes[s]["position"]) for s in node_list], dtype=np.int64
        ).reshape(-1, 3)
        edge_list = list(self.edges)
        edges = (
            np.array(edge_list, dtype=segmentation_dtype).reshape(-1, 2)
            if edge_list
            else np.empty((0, 2), dtype=segmentation_dtype)
        )
        affinities = np.array(
            [float(self.edges[u, v].get("affinity", 0.0)) for u, v in edge_list],
            dtype=np.float32,
        )
        return AgglomerateGraphData(
            segments=segments,
            edges=edges,
            positions=positions,
            affinities=affinities,
        )

    def __repr__(self) -> str:
        return f"AgglomerateGraph({len(self.nodes)} nodes, {len(self.edges)} edges)"


class AgglomerateAttachment(Attachment):
    data_format: Literal[AttachmentDataFormat.Zarr3, AttachmentDataFormat.HDF5]
    container_name = "agglomerates"
    type_name = "agglomerate"

    @classmethod
    def create_and_add_to(
        cls, layer: SegmentationLayer, name: str, graph: AgglomerateGraph
    ) -> AgglomerateAttachment:
        """Create a Zarr v3 agglomerate attachment from a networkx graph and add it to a segmentation layer.

        `name` is the attachment name.

        The `graph` must have:
        - Integer node labels (segment IDs, 1-based)
        - 'position' node attribute: Vec3Int (x, y, z)
        - 'affinity' edge attribute: float

        Connected components become agglomerates.

        Returns the added AgglomerateAttachment.
        """
        assert layer.dtype in (np.uint32, np.uint64), (
            f"Cannot create agglomerate attachment for segmentation layer with dtype {layer.dtype}. "
            + "Please use a segmentation layer with dtype uint32 or uint64."
        )
        attachment_path = layer.resolved_path / cls.container_name / name
        if attachment_path.exists():
            raise FileExistsError(
                f"Agglomerate attachment at path {attachment_path} already exists."
            )
        attachment = cls.create(attachment_path, graph, segmentation_dtype=layer.dtype)
        layer.attachments._add_attachment(attachment)
        return attachment

    @classmethod
    def create(
        cls,
        path: str | PathLike | UPath,
        graph: AgglomerateGraph,
        *,
        segmentation_dtype: np.typing.DTypeLike = "uint32",
    ) -> AgglomerateAttachment:
        """Create and write a Zarr v3 agglomerate attachment from a networkx graph.

        The graph must have:
        - Integer node labels (segment IDs, 1-based)
        - 'position' node attribute: Vec3Int (x, y, z)
        - 'affinity' edge attribute: float

        Connected components become agglomerates.

        The AgglomerateAttachment will be created at `path`.

        `segmentation_dtype` must match the dtype of the corresponding segmentation layer.

        Returns an AgglomerateAttachment usable directly with
        `seg_layer.attachments.add_attachment_as_copy(attachment)`.
        """
        segmentation_dtype = np.dtype(segmentation_dtype)

        path = UPath(path)
        path.mkdir(parents=True, exist_ok=True)

        n_segments, _ = _validate_segment_ids(graph)

        if segmentation_dtype not in (np.uint32, np.uint64):
            raise ValueError("Segmentation dtype must be uint32 or uint64.")

        # Identify connected components → agglomerates
        agglomerates: list[set[int]] = list(nx.connected_components(graph))
        # Sort by minimum segment ID in each component
        agglomerates.sort(key=lambda c: min(c))
        n_agglomerates = len(agglomerates)

        # Build segment_to_agglomerate: index 0 = background = 0
        segment_to_agglomerate = np.zeros(n_segments + 1, dtype=np.uint64)
        for agg_idx, agglomerate in enumerate(agglomerates):
            agg_id = agg_idx + 1  # 1-based
            for seg_id in agglomerate:
                segment_to_agglomerate[seg_id] = agg_id

        # For each agglomerate, sort segments ascending
        sorted_segments_per_agg = [sorted(c) for c in agglomerates]

        # Build agglomerate_to_segments, agglomerate_to_segments_offsets, agglomerate_to_positions
        # offsets[0] = offsets[1] = 0  (agglomerate 0 is reserved/empty)
        # Shape: (n_agglomerates + 2,) because agg IDs are 0..n_agglomerates
        # and CSR needs one extra sentinel. The first two entries are 0.
        agglomerate_to_segments_offsets = np.zeros(n_agglomerates + 2, dtype=np.uint64)
        agglomerate_to_segments = np.zeros((n_segments,), dtype=segmentation_dtype)
        agglomerate_to_positions = np.zeros((n_segments, 3), dtype=np.int32)
        i = 0
        for agg_idx, agg_seg_ids in enumerate(sorted_segments_per_agg):
            agg_id = agg_idx + 1  # 1-based
            # Writing the next agglomerate's offset
            agglomerate_to_segments_offsets[agg_id + 1] = (
                agglomerate_to_segments_offsets[agg_id] + len(agg_seg_ids)
            )
            for seg_id in agg_seg_ids:
                agglomerate_to_segments[i] = seg_id
                seg_position = graph.nodes[seg_id]["position"]
                agglomerate_to_positions[i] = seg_position.to_np()
                i += 1

        # Build edges with local indices
        # Shape: (n_agglomerates + 2,) — same rationale as agglomerate_to_segments_offsets
        n_edges = len(graph.edges)
        agglomerate_to_edges_offsets = np.zeros(n_agglomerates + 2, dtype=np.uint64)
        agglomerate_to_edges = np.zeros((n_edges, 2), dtype=segmentation_dtype)
        agglomerate_to_affinities = np.zeros((n_edges,), dtype=np.float32)
        i = 0

        for agg_idx, agg_seg_ids in enumerate(sorted_segments_per_agg):
            agg_id = agg_idx + 1  # 1-based
            # Map global segment ID → local index within this agglomerate
            local_index = {seg_id: i for i, seg_id in enumerate(agg_seg_ids)}
            agg_edges: list[tuple[int, int, float]] = []
            for u, v, data in graph.edges(agg_seg_ids, data=True):
                # Only include edges where both endpoints are in this agglomerate
                if u not in local_index or v not in local_index:
                    continue
                local_seg_id_u = local_index[u]
                local_seg_id_v = local_index[v]
                if local_seg_id_u > local_seg_id_v:
                    local_seg_id_u, local_seg_id_v = local_seg_id_v, local_seg_id_u
                affinity = float(data.get("affinity", 0.0))
                agg_edges.append((local_seg_id_u, local_seg_id_v, affinity))
            # Sort lexicographically by (local_seg_id_u, local_seg_id_v)
            agg_edges.sort(key=lambda e: (e[0], e[1]))
            # Writing the next agglomerate's offset
            agglomerate_to_edges_offsets[agg_id + 1] = agglomerate_to_edges_offsets[
                agg_id
            ] + len(agg_edges)
            for local_seg_id_u, local_seg_id_v, affinity in agg_edges:
                agglomerate_to_edges[i] = (local_seg_id_u, local_seg_id_v)
                agglomerate_to_affinities[i] = affinity
                i += 1

        # Write group zarr.json
        group_meta = {
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {
                "voxelytics": {
                    "artifact_schema_version": 4,
                    "artifact_class": "AgglomerateViewArtifact",
                }
            },
        }
        (path / "zarr.json").write_text(json.dumps(group_meta, indent=2))

        # Target sizes from the spec
        OFFSET_CHUNK = 64 * 1024  # 64 KB
        OFFSET_SHARD = 256 * 1024 * 1024  # 256 MB
        DATA_CHUNK = 256 * 1024  # 256 KB
        DATA_SHARD = 1024 * 1024 * 1024  # 1 GB

        write_zarr3_array(
            path / "segment_to_agglomerate",
            segment_to_agglomerate,
            target_chunk_size_bytes=DATA_CHUNK,
            target_shard_size_bytes=DATA_SHARD,
        )
        write_zarr3_array(
            path / "agglomerate_to_segments_offsets",
            agglomerate_to_segments_offsets,
            target_chunk_size_bytes=OFFSET_CHUNK,
            target_shard_size_bytes=OFFSET_SHARD,
        )
        write_zarr3_array(
            path / "agglomerate_to_segments",
            agglomerate_to_segments,
            target_chunk_size_bytes=DATA_CHUNK,
            target_shard_size_bytes=DATA_SHARD,
        )
        write_zarr3_array(
            path / "agglomerate_to_edges_offsets",
            agglomerate_to_edges_offsets,
            target_chunk_size_bytes=OFFSET_CHUNK,
            target_shard_size_bytes=OFFSET_SHARD,
        )
        write_zarr3_array(
            path / "agglomerate_to_edges",
            agglomerate_to_edges,
            target_chunk_size_bytes=DATA_CHUNK,
            target_shard_size_bytes=DATA_SHARD,
        )
        write_zarr3_array(
            path / "agglomerate_to_affinities",
            agglomerate_to_affinities,
            target_chunk_size_bytes=DATA_CHUNK,
            target_shard_size_bytes=DATA_SHARD,
        )
        write_zarr3_array(
            path / "agglomerate_to_positions",
            agglomerate_to_positions,
            target_chunk_size_bytes=DATA_CHUNK,
            target_shard_size_bytes=DATA_SHARD,
        )

        return cls.from_path_and_name(
            path,
            path.name,
            data_format=AttachmentDataFormat.Zarr3,
        )

    def to_graph(self) -> AgglomerateGraph:
        """Read the agglomerate attachment from disk and reconstruct an AgglomerateGraph.

        Returns an AgglomerateGraph with:
        - Integer node labels (segment IDs, 1-based)
        - 'position' node attribute: Vec3Int (x, y, z)
        - 'affinity' edge attribute: float
        """
        if self.data_format != AttachmentDataFormat.Zarr3:
            raise NotImplementedError(
                f"to_graph() only supports Zarr3 format, got {self.data_format}"
            )

        agglomerate_to_segments = read_zarr3_array(
            self.path / "agglomerate_to_segments"
        )
        agglomerate_to_segments_offsets = read_zarr3_array(
            self.path / "agglomerate_to_segments_offsets"
        )
        agglomerate_to_positions = read_zarr3_array(
            self.path / "agglomerate_to_positions"
        )
        agglomerate_to_edges = read_zarr3_array(self.path / "agglomerate_to_edges")
        agglomerate_to_edges_offsets = read_zarr3_array(
            self.path / "agglomerate_to_edges_offsets"
        )
        agglomerate_to_affinities = read_zarr3_array(
            self.path / "agglomerate_to_affinities"
        )

        graph = AgglomerateGraph()

        # agglomerate_to_segments_offsets shape is (n_agglomerates + 2,): indices 0..n_agglomerates+1
        n_agglomerates = len(agglomerate_to_segments_offsets) - 2

        for agg_id in range(1, n_agglomerates + 1):
            seg_start = int(agglomerate_to_segments_offsets[agg_id])
            seg_end = int(agglomerate_to_segments_offsets[agg_id + 1])
            agg_seg_ids = agglomerate_to_segments[seg_start:seg_end]
            agg_seg_positions = agglomerate_to_positions[seg_start:seg_end]

            for local_idx in range(len(agg_seg_ids)):
                seg_id = int(agg_seg_ids[local_idx])
                pos = agg_seg_positions[local_idx]
                graph.add_segment(seg_id, position=Vec3Int(pos))

            edge_start = int(agglomerate_to_edges_offsets[agg_id])
            edge_end = int(agglomerate_to_edges_offsets[agg_id + 1])
            for edge_idx in range(edge_start, edge_end):
                local_seg_id_u = int(agglomerate_to_edges[edge_idx, 0])
                local_seg_id_v = int(agglomerate_to_edges[edge_idx, 1])
                seg_id_u = int(agg_seg_ids[local_seg_id_u])
                seg_id_v = int(agg_seg_ids[local_seg_id_v])
                graph.add_affinity_edge(
                    seg_id_u,
                    seg_id_v,
                    affinity=float(agglomerate_to_affinities[edge_idx]),
                )

        return graph
