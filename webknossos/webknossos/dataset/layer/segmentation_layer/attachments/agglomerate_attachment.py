from __future__ import annotations

import json
from os import PathLike
from typing import TYPE_CHECKING, Literal

import networkx as nx
import numpy as np
from upath import UPath

if TYPE_CHECKING:
    from webknossos.proofreading.agglomerate_graph import AgglomerateGraphData

from webknossos.dataset_properties import AttachmentDataFormat
from webknossos.geometry import Vec3Int

from ._utils import read_zarr3_array, write_zarr3_array
from .attachment import Attachment


class AgglomerateGraph(nx.Graph):
    """Typed networkx Graph for agglomerate data.
    Node labels are segment IDs (integers, 1-based).
    """

    def add_segment(self, segment_id: int, position: Vec3Int) -> None:
        """Add a segment node with a (x, y, z) position."""
        self.add_node(segment_id, position=position)

    def add_affinity_edge(self, u: int, v: int, affinity: float) -> None:
        """Add an undirected edge between two segment IDs with an affinity score."""
        self.add_edge(u, v, affinity=affinity)

    def to_agglomerate_graph_data(self) -> "AgglomerateGraphData":
        """Convert to an AgglomerateGraphData (flat numpy-array representation).

        Returns an AgglomerateGraphData with:
        - segments: segment IDs, dtype uint64
        - positions: [x, y, z] per segment, dtype int64
        - edges: [source, target] segment ID pairs, dtype uint64
        - affinities: affinity per edge, dtype float32
        """
        from webknossos.proofreading.agglomerate_graph import AgglomerateGraphData

        node_list = list(self.nodes)
        segments = np.array(node_list, dtype=np.uint64)
        positions = np.array(
            [list(self.nodes[s]["position"]) for s in node_list], dtype=np.int64
        ).reshape(-1, 3)
        edge_list = list(self.edges)
        edges = (
            np.array(edge_list, dtype=np.uint64).reshape(-1, 2)
            if edge_list
            else np.empty((0, 2), dtype=np.uint64)
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


class AgglomerateAttachment(Attachment):
    data_format: Literal[AttachmentDataFormat.Zarr3, AttachmentDataFormat.HDF5]
    container_name = "agglomerates"
    type_name = "agglomerate"

    @classmethod
    def create(
        cls,
        path: str | PathLike | UPath,
        graph: nx.Graph,
    ) -> "AgglomerateAttachment":
        """Create and write a Zarr v3 agglomerate attachment from a networkx graph.

        The graph must have:
        - Integer node labels (segment IDs, 1-based)
        - 'position' node attribute: Vec3Int (x, y, z)
        - 'affinity' edge attribute: float

        Connected components become agglomerates.

        `path` is the mapping group directory (e.g. `…/agglomerate_view_75`).
        The directory name is used as the attachment name.

        Returns an AgglomerateAttachment usable directly with
        `seg_layer.attachments.add_attachment_as_copy(attachment)`.
        """
        path = UPath(path)
        path.mkdir(parents=True, exist_ok=True)

        # --- Validate and collect segment IDs ---
        segment_ids = list(graph.nodes)
        if not segment_ids:
            raise ValueError("Graph has no nodes.")
        for sid in segment_ids:
            if not isinstance(sid, int) or sid < 1:
                raise ValueError(
                    f"All node labels must be positive integers (1-based), got {sid!r}"
                )

        n_segments = max(segment_ids)
        if set(segment_ids) != set(range(1, n_segments + 1)):
            missing = sorted(set(range(1, n_segments + 1)) - set(segment_ids))
            raise ValueError(
                f"Segment IDs must be dense (1..{n_segments}) with no gaps, "
                f"missing: {missing}"
            )
        segmentation_dtype = np.uint32 if n_segments < 2**32 else np.uint64
        seg_dtype_str = "uint32" if n_segments < 2**32 else "uint64"

        # --- Identify connected components → agglomerates ---
        components = list(nx.connected_components(graph))
        # Sort by minimum segment ID in each component
        components.sort(key=lambda c: min(c))
        n_agglomerates = len(components)

        # Build segment_to_agglomerate: index 0 = background = 0
        segment_to_agglomerate = np.zeros(n_segments + 1, dtype=np.uint64)
        for agg_idx, component in enumerate(components):
            agg_id = agg_idx + 1  # 1-based
            for seg_id in component:
                segment_to_agglomerate[seg_id] = agg_id

        # For each agglomerate, sort segments ascending
        sorted_segments_per_agg = [sorted(c) for c in components]

        # Build agglomerate_to_segments and agglomerate_to_segments_offsets
        # offsets[0] = offsets[1] = 0  (agglomerate 0 is reserved/empty)
        # Shape: (n_agglomerates + 2,) because agg IDs are 0..n_agglomerates
        # and CSR needs one extra sentinel. The first two entries are 0.
        seg_offsets = np.zeros(n_agglomerates + 2, dtype=np.uint64)
        segments_list = []
        positions_list = []
        for agg_idx, segs in enumerate(sorted_segments_per_agg):
            seg_offsets[agg_idx + 2] = seg_offsets[agg_idx + 1] + len(segs)
            for seg_id in segs:
                segments_list.append(seg_id)
                pos = graph.nodes[seg_id]["position"]
                positions_list.append([int(pos[0]), int(pos[1]), int(pos[2])])

        agglomerate_to_segments = np.array(segments_list, dtype=segmentation_dtype)
        agglomerate_to_positions = np.array(positions_list, dtype=np.int32)

        # Build edges with local indices
        # Shape: (n_agglomerates + 2,) — same rationale as seg_offsets
        edge_offsets = np.zeros(n_agglomerates + 2, dtype=np.uint64)
        edges_list = []
        affinities_list = []

        for agg_idx, segs in enumerate(sorted_segments_per_agg):
            # Map global segment ID → local index within this agglomerate
            local_index = {seg_id: i for i, seg_id in enumerate(segs)}
            agg_edges = []
            for u, v, data in graph.edges(segs, data=True):
                # Only include edges where both endpoints are in this agglomerate
                if u not in local_index or v not in local_index:
                    continue
                n1 = local_index[u]
                n2 = local_index[v]
                if n1 > n2:
                    n1, n2 = n2, n1
                affinity = float(data.get("affinity", 0.0))
                agg_edges.append((n1, n2, affinity))
            # Sort lexicographically by (n1, n2)
            agg_edges.sort(key=lambda e: (e[0], e[1]))
            edge_offsets[agg_idx + 2] = edge_offsets[agg_idx + 1] + len(agg_edges)
            for n1, n2, affinity in agg_edges:
                edges_list.append([n1, n2])
                affinities_list.append(affinity)

        n_edges = len(edges_list)
        agglomerate_to_edges = (
            np.array(edges_list, dtype=segmentation_dtype).reshape(n_edges, 2)
            if n_edges > 0
            else np.empty((0, 2), dtype=segmentation_dtype)
        )
        agglomerate_to_affinities = np.array(affinities_list, dtype=np.float32)

        # --- Write group zarr.json ---
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
            dtype="uint64",
            target_chunk_size_bytes=DATA_CHUNK,
            target_shard_size_bytes=DATA_SHARD,
        )
        write_zarr3_array(
            path / "agglomerate_to_segments_offsets",
            seg_offsets,
            dtype="uint64",
            target_chunk_size_bytes=OFFSET_CHUNK,
            target_shard_size_bytes=OFFSET_SHARD,
        )
        write_zarr3_array(
            path / "agglomerate_to_segments",
            agglomerate_to_segments,
            dtype=seg_dtype_str,
            target_chunk_size_bytes=DATA_CHUNK,
            target_shard_size_bytes=DATA_SHARD,
        )
        write_zarr3_array(
            path / "agglomerate_to_edges_offsets",
            edge_offsets,
            dtype="uint64",
            target_chunk_size_bytes=OFFSET_CHUNK,
            target_shard_size_bytes=OFFSET_SHARD,
        )
        write_zarr3_array(
            path / "agglomerate_to_edges",
            agglomerate_to_edges,
            dtype=seg_dtype_str,
            target_chunk_size_bytes=DATA_CHUNK,
            target_shard_size_bytes=DATA_SHARD,
        )
        write_zarr3_array(
            path / "agglomerate_to_affinities",
            agglomerate_to_affinities,
            dtype="float32",
            target_chunk_size_bytes=DATA_CHUNK,
            target_shard_size_bytes=DATA_SHARD,
        )
        write_zarr3_array(
            path / "agglomerate_to_positions",
            agglomerate_to_positions,
            dtype="int32",
            target_chunk_size_bytes=DATA_CHUNK,
            target_shard_size_bytes=DATA_SHARD,
        )

        return cls.from_path_and_name(
            path,
            path.name,
            data_format=AttachmentDataFormat.Zarr3,
        )

    def to_graph(self) -> "AgglomerateGraph":
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

        segments = read_zarr3_array(self.path / "agglomerate_to_segments")
        seg_offsets = read_zarr3_array(self.path / "agglomerate_to_segments_offsets")
        positions = read_zarr3_array(self.path / "agglomerate_to_positions")
        edges = read_zarr3_array(self.path / "agglomerate_to_edges")
        edge_offsets = read_zarr3_array(self.path / "agglomerate_to_edges_offsets")
        affinities = read_zarr3_array(self.path / "agglomerate_to_affinities")

        graph = AgglomerateGraph()

        # seg_offsets shape is (n_agglomerates + 2,): indices 0..n_agglomerates+1
        n_agglomerates = len(seg_offsets) - 2

        for agg_id in range(1, n_agglomerates + 1):
            seg_start = int(seg_offsets[agg_id])
            seg_end = int(seg_offsets[agg_id + 1])
            agg_segs = segments[seg_start:seg_end]
            agg_positions = positions[seg_start:seg_end]

            for local_idx in range(len(agg_segs)):
                seg_id = int(agg_segs[local_idx])
                pos = agg_positions[local_idx]
                graph.add_segment(
                    seg_id, Vec3Int(int(pos[0]), int(pos[1]), int(pos[2]))
                )

            edge_start = int(edge_offsets[agg_id])
            edge_end = int(edge_offsets[agg_id + 1])
            for edge_idx in range(edge_start, edge_end):
                local_n1 = int(edges[edge_idx, 0])
                local_n2 = int(edges[edge_idx, 1])
                global_u = int(agg_segs[local_n1])
                global_v = int(agg_segs[local_n2])
                graph.add_affinity_edge(global_u, global_v, float(affinities[edge_idx]))

        return graph
