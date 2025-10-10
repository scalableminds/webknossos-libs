import numpy as np

from webknossos.proofreading import agglomerate_graph_pb2


class AgglomerateGraph:
    """Represents an agglomerate graph.

    Attributes:
        segments (np.ndarray): A numpy array of segment ids.
        edges (np.ndarray): A numpy array of edges.
        positions (np.ndarray): A numpy array of positions.
        affinities (np.ndarray): A numpy array of affinities.
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
    ) -> "AgglomerateGraph":
        """Create an AgglomerateGraph from a protobuf binary.

        Args:
            protobuf_binary (bytes): The protobuf binary to create the AgglomerateGraph from.

        Returns:
            AgglomerateGraph
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
