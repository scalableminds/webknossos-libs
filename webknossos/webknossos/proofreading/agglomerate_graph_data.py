from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from webknossos.proofreading.generated import agglomerate_graph_pb2

if TYPE_CHECKING:
    from webknossos.dataset.layer.segmentation_layer.attachments.agglomerate_attachment import (
        AgglomerateGraph,
    )


class AgglomerateGraphData:
    """Flat numpy-array representation of an agglomerate graph, as returned by the proofreading API.

    Attributes:
        segments (np.ndarray): Segment IDs, shape (N,), dtype uint64.
        edges (np.ndarray): Edge pairs [source, target] as segment IDs, shape (E, 2), dtype uint64.
        positions (np.ndarray): Positions [x, y, z] for each segment, shape (N, 3), dtype int64.
        affinities (np.ndarray): Affinity score for each edge, shape (E,), dtype float32.
    """

    def __init__(
        self,
        segments: np.ndarray,
        edges: np.ndarray,
        positions: np.ndarray,
        affinities: np.ndarray,
    ) -> None:
        self.segments = segments
        self.edges = edges
        self.positions = positions
        self.affinities = affinities

    @classmethod
    def from_proto(
        cls, agglomerate_graph_proto: agglomerate_graph_pb2.AgglomerateGraph
    ) -> AgglomerateGraphData:
        """Create an AgglomerateGraphData from a protobuf message.

        Returns:
            AgglomerateGraphData
        """

        return cls(
            segments=np.array(agglomerate_graph_proto.segments, dtype=np.uint64),
            edges=np.array(
                [[edge.source, edge.target] for edge in agglomerate_graph_proto.edges],
                dtype=np.uint64,
            ),
            positions=np.array(
                [[p.x, p.y, p.z] for p in agglomerate_graph_proto.positions],
                dtype=np.int64,
            ),
            affinities=np.array(agglomerate_graph_proto.affinities, dtype=np.float32),
        )

    def to_agglomerate_graph(self) -> AgglomerateGraph:
        """Convert to a networkx-based AgglomerateGraph.

        Returns an AgglomerateGraph (nx.Graph subclass) with:
        - Integer node labels (segment IDs)
        - 'position' node attribute: Vec3Int (x, y, z)
        - 'affinity' edge attribute: float
        """
        from webknossos.dataset.layer.segmentation_layer.attachments.agglomerate_attachment import (
            AgglomerateGraph,
        )
        from webknossos.geometry import Vec3Int

        graph: AgglomerateGraph = AgglomerateGraph()
        for i, seg_id in enumerate(self.segments):
            pos = self.positions[i]
            graph.add_segment(int(seg_id), position=Vec3Int(pos))
        for i, (src, tgt) in enumerate(self.edges):
            graph.add_affinity_edge(
                int(src), int(tgt), affinity=float(self.affinities[i])
            )
        return graph
