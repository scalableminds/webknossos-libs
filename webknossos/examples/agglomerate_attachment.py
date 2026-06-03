"""Create an agglomerate attachment and add it to a segmentation layer.

An agglomerate attachment stores a supervoxel graph: which segments belong
together (agglomerates), the edges between them, the affinity scores of those
edges, and a representative position for each segment.

This example builds a small synthetic segmentation dataset, constructs an
AgglomerateGraph from scratch, writes it to disk with
AgglomerateAttachment.create(), and registers it on the segmentation layer.
"""

import numpy as np

import webknossos as wk
from webknossos import AgglomerateAttachment, AgglomerateGraph
from webknossos.geometry import Vec3Int


def main() -> None:
    # -------------------------------------------------------------------------
    # 1. Create a minimal segmentation dataset
    # -------------------------------------------------------------------------
    dataset = wk.Dataset("testoutput/agglomerate_example", voxel_size=(8, 8, 8))
    seg_layer = dataset.add_layer(
        layer_name="segmentation",
        category="segmentation",
        dtype="uint32",
        largest_segment_id=7,
    ).as_segmentation_layer()
    mag1 = seg_layer.add_mag("1")

    # Paint segment IDs 1‥7 into a small volume so WEBKNOSSOS can look them up.
    # Segment IDs must be dense: 1, 2, …, n with no gaps.
    # Shape is (channels=1, x=2, y=2, z=2).
    voxel_data = np.array([[[[1, 5], [3, 7]], [[2, 6], [4, 1]]]], dtype=np.uint32)
    mag1.write(data=voxel_data, allow_resize=True)

    # -------------------------------------------------------------------------
    # 2. Build an AgglomerateGraph
    #
    # Each node is a segment ID (1-based integer).
    # Each node carries a representative voxel position (x, y, z).
    # Edges connect adjacent segments; the affinity score indicates how
    # strongly the supervoxel merger evidence supports joining them.
    # -------------------------------------------------------------------------
    graph = AgglomerateGraph()

    # Segment positions (representative voxels)
    graph.add_segment(1, position=Vec3Int(10, 20, 30))
    graph.add_segment(2, position=Vec3Int(15, 25, 35))
    graph.add_segment(3, position=Vec3Int(50, 60, 70))
    graph.add_segment(4, position=Vec3Int(55, 65, 75))
    graph.add_segment(5, position=Vec3Int(100, 110, 120))
    graph.add_segment(6, position=Vec3Int(105, 115, 125))
    graph.add_segment(7, position=Vec3Int(200, 210, 220))

    # Edges and their affinity scores.
    # Segments connected by edges (directly or transitively) form one agglomerate.
    # Here we create two agglomerates:
    #   • agglomerate A: segments {1, 2, 3, 4, 7}  (all connected)
    #   • agglomerate B: segments {5, 6}
    graph.add_affinity_edge(1, 2, affinity=0.92)
    graph.add_affinity_edge(2, 3, affinity=0.85)
    graph.add_affinity_edge(3, 4, affinity=0.78)
    graph.add_affinity_edge(1, 7, affinity=0.65)
    graph.add_affinity_edge(5, 6, affinity=0.88)

    # -------------------------------------------------------------------------
    # 3. Write the attachment to disk and register it on the layer
    #
    # create_and_add_to() creates an AgglomerateAttachment object and adds it to
    # the segmentation layer. The attachment name becomes the attachment name
    # that WEBKNOSSOS displays in the UI.
    # -------------------------------------------------------------------------
    attachment = AgglomerateAttachment.create_and_add_to(
        seg_layer, "agglomerate_view_75", graph
    )

    print(f"Created agglomerate attachment: {attachment}")

    # -------------------------------------------------------------------------
    # 4. Round-trip: read the attachment back as a graph
    # -------------------------------------------------------------------------
    recovered_graph = attachment.to_graph()

    print(
        f"\nRecovered graph: {recovered_graph.number_of_nodes()} nodes, "
        f"{recovered_graph.number_of_edges()} edges"
    )
    for node, data in sorted(recovered_graph.nodes(data=True)):
        print(f"  segment {node}: position={data['position']}")


if __name__ == "__main__":
    main()
