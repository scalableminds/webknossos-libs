"""Tests for AgglomerateAttachment.create() — covers the worked example from agglomerate.md."""

import shutil
import tempfile
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest
from upath import UPath

from webknossos import AgglomerateAttachment, AgglomerateGraph
from webknossos.dataset.layer.segmentation_layer.attachments._utils import (
    read_zarr3_array,
)
from webknossos.geometry import Vec3Int


def read_zarr_array(path: Path) -> np.ndarray:  # type: ignore[type-arg]
    return read_zarr3_array(UPath(path))


@pytest.fixture()
def tmp_dir() -> Generator[Path, None, None]:
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)


def build_worked_example_graph() -> AgglomerateGraph:
    """Build the graph from the agglomerate.md worked example.

    edges = [[1,2], [2,3], [3,4], [5,6], [1,7]]
    segments 1-4,7 → agglomerate 1; segments 5,6 → agglomerate 2

    Affinities assigned to match expected edge order after sorting:
      agg1 sorted edges (local): (0,1)=1-2, (0,4)=1-7, (1,2)=2-3, (2,3)=3-4
      → affinities: 124.0, 65.5, 0.0, 250.5
    """
    G = AgglomerateGraph()
    # Positions: (seg_id * 10, seg_id * 20, seg_id * 30) for uniqueness
    for i in range(1, 8):
        G.add_segment(i, Vec3Int(i * 10, i * 20, i * 30))
    G.add_affinity_edge(1, 2, affinity=124.0)
    G.add_affinity_edge(2, 3, affinity=0.0)
    G.add_affinity_edge(3, 4, affinity=250.5)
    G.add_affinity_edge(5, 6, affinity=80.0)
    G.add_affinity_edge(1, 7, affinity=65.5)
    return G


def test_build_worked_example(tmp_dir: Path) -> None:
    mapping_path = tmp_dir / "agglomerate_view_75"
    G = build_worked_example_graph()
    attachment = AgglomerateAttachment.create(mapping_path, G)

    # --- Check returned attachment ---
    assert attachment.name == "agglomerate_view_75"
    from webknossos.dataset_properties import AttachmentDataFormat

    assert attachment.data_format == AttachmentDataFormat.Zarr3

    # --- Check zarr.json ---
    import json

    meta = json.loads((mapping_path / "zarr.json").read_text())
    assert meta["zarr_format"] == 3
    assert meta["node_type"] == "group"
    assert meta["attributes"]["voxelytics"]["artifact_schema_version"] == 4
    assert (
        meta["attributes"]["voxelytics"]["artifact_class"] == "AgglomerateViewArtifact"
    )

    # --- segment_to_agglomerate ---
    # [0, 1, 1, 1, 1, 2, 2, 1]  shape (8,)
    s2a = read_zarr_array(mapping_path / "segment_to_agglomerate")
    expected_s2a = np.array([0, 1, 1, 1, 1, 2, 2, 1], dtype=np.uint64)
    np.testing.assert_array_equal(s2a, expected_s2a)

    # --- agglomerate_to_segments_offsets ---
    # [0, 0, 5, 7]  shape (4,)
    off = read_zarr_array(mapping_path / "agglomerate_to_segments_offsets")
    expected_off = np.array([0, 0, 5, 7], dtype=np.uint64)
    np.testing.assert_array_equal(off, expected_off)

    # --- agglomerate_to_segments ---
    # [1, 2, 3, 4, 7, 5, 6]  shape (7,)
    segs = read_zarr_array(mapping_path / "agglomerate_to_segments")
    expected_segs = np.array([1, 2, 3, 4, 7, 5, 6], dtype=np.uint32)
    np.testing.assert_array_equal(segs, expected_segs)

    # --- agglomerate_to_edges_offsets ---
    # [0, 0, 4, 5]  shape (4,)
    eoff = read_zarr_array(mapping_path / "agglomerate_to_edges_offsets")
    expected_eoff = np.array([0, 0, 4, 5], dtype=np.uint64)
    np.testing.assert_array_equal(eoff, expected_eoff)

    # --- agglomerate_to_edges ---
    # [[0,1],[0,4],[1,2],[2,3],[0,1]]  shape (5,2)
    edges = read_zarr_array(mapping_path / "agglomerate_to_edges")
    expected_edges = np.array([[0, 1], [0, 4], [1, 2], [2, 3], [0, 1]], dtype=np.uint32)
    np.testing.assert_array_equal(edges, expected_edges)

    # --- agglomerate_to_affinities ---
    # [124.0, 65.5, 0.0, 250.5, 80.0]  shape (5,)
    affs = read_zarr_array(mapping_path / "agglomerate_to_affinities")
    expected_affs = np.array([124.0, 65.5, 0.0, 250.5, 80.0], dtype=np.float32)
    np.testing.assert_array_almost_equal(affs, expected_affs)

    # --- agglomerate_to_positions ---
    # Co-indexed with agglomerate_to_segments: [1,2,3,4,7,5,6]
    positions = read_zarr_array(mapping_path / "agglomerate_to_positions")
    assert positions.shape == (7, 3)
    assert positions.dtype == np.int32
    seg_order = [1, 2, 3, 4, 7, 5, 6]
    for local_idx, seg_id in enumerate(seg_order):
        expected_pos = np.array([seg_id * 10, seg_id * 20, seg_id * 30], dtype=np.int32)
        np.testing.assert_array_equal(positions[local_idx], expected_pos)


def test_build_single_node_no_edges(tmp_dir: Path) -> None:
    """Single segment, no edges — produces one agglomerate with one segment and zero edges."""
    G = AgglomerateGraph()
    G.add_segment(1, Vec3Int(5, 10, 15))
    attachment = AgglomerateAttachment.create(tmp_dir / "agg_view_1", G)
    assert attachment.name == "agg_view_1"

    s2a = read_zarr_array(tmp_dir / "agg_view_1" / "segment_to_agglomerate")
    np.testing.assert_array_equal(s2a, [0, 1])

    off = read_zarr_array(tmp_dir / "agg_view_1" / "agglomerate_to_segments_offsets")
    np.testing.assert_array_equal(off, [0, 0, 1])

    segs = read_zarr_array(tmp_dir / "agg_view_1" / "agglomerate_to_segments")
    np.testing.assert_array_equal(segs, [1])

    eoff = read_zarr_array(tmp_dir / "agg_view_1" / "agglomerate_to_edges_offsets")
    np.testing.assert_array_equal(eoff, [0, 0, 0])

    edges = read_zarr_array(tmp_dir / "agg_view_1" / "agglomerate_to_edges")
    assert edges.shape == (0, 2)

    affs = read_zarr_array(tmp_dir / "agg_view_1" / "agglomerate_to_affinities")
    assert affs.shape == (0,)


def test_build_invalid_segment_id(tmp_dir: Path) -> None:
    G = AgglomerateGraph()
    G.add_node(0, position=Vec3Int(1, 2, 3))  # 0 is invalid (background)
    with pytest.raises(ValueError, match="positive integers"):
        AgglomerateAttachment.create(tmp_dir / "agg_invalid", G)


def test_build_sparse_segment_ids_rejected(tmp_dir: Path) -> None:
    G = AgglomerateGraph()
    G.add_segment(1, Vec3Int(1, 2, 3))
    G.add_segment(3, Vec3Int(4, 5, 6))  # gap: segment 2 is missing
    with pytest.raises(ValueError, match="dense"):
        AgglomerateAttachment.create(tmp_dir / "agg_sparse", G)


def test_agglomerate_graph_api() -> None:
    G = AgglomerateGraph()
    G.add_segment(1, Vec3Int(10, 20, 30))
    G.add_segment(2, Vec3Int(15, 25, 35))
    G.add_affinity_edge(1, 2, affinity=0.9)
    assert G.number_of_nodes() == 2
    assert G.number_of_edges() == 1
    assert G.nodes[1]["position"] == Vec3Int(10, 20, 30)
    assert G.edges[1, 2]["affinity"] == pytest.approx(0.9)


def assert_graphs_equal(G1: AgglomerateGraph, G2: AgglomerateGraph) -> None:
    """Compare two AgglomerateGraphs for semantic equality (float32 affinity precision)."""
    assert set(G1.nodes) == set(G2.nodes), (
        f"Node sets differ: {set(G1.nodes)} vs {set(G2.nodes)}"
    )
    for node in G1.nodes:
        assert G1.nodes[node]["position"] == G2.nodes[node]["position"], (
            f"Position mismatch for node {node}: "
            f"{G1.nodes[node]['position']} vs {G2.nodes[node]['position']}"
        )
    edges1 = {frozenset(e): G1.edges[e]["affinity"] for e in G1.edges}
    edges2 = {frozenset(e): G2.edges[e]["affinity"] for e in G2.edges}
    assert set(edges1.keys()) == set(edges2.keys()), (
        f"Edge sets differ: {set(edges1.keys())} vs {set(edges2.keys())}"
    )
    for edge_key in edges1:
        assert np.float32(edges1[edge_key]) == pytest.approx(
            np.float32(edges2[edge_key])
        ), (
            f"Affinity mismatch for edge {edge_key}: {edges1[edge_key]} vs {edges2[edge_key]}"
        )


def test_roundtrip_worked_example(tmp_dir: Path) -> None:
    """create() followed by to_graph() reproduces the original graph exactly."""
    G = build_worked_example_graph()
    attachment = AgglomerateAttachment.create(tmp_dir / "agglomerate_view_75", G)
    G2 = attachment.to_graph()
    assert_graphs_equal(G, G2)


def test_roundtrip_single_node(tmp_dir: Path) -> None:
    """Roundtrip with a single isolated node (no edges)."""
    G = AgglomerateGraph()
    G.add_segment(1, Vec3Int(100, 200, 300))
    attachment = AgglomerateAttachment.create(tmp_dir / "agg_view_single", G)
    G2 = attachment.to_graph()
    assert_graphs_equal(G, G2)


def test_roundtrip_multiple_components(tmp_dir: Path) -> None:
    """Roundtrip with several disconnected components (dense IDs required)."""
    G = AgglomerateGraph()
    # Component A: 1-2-3
    G.add_segment(1, Vec3Int(1, 2, 3))
    G.add_segment(2, Vec3Int(4, 5, 6))
    G.add_segment(3, Vec3Int(7, 8, 9))
    G.add_affinity_edge(1, 2, affinity=0.75)
    G.add_affinity_edge(2, 3, affinity=0.5)
    # Component B: isolated node (dense: 4 follows 3)
    G.add_segment(4, Vec3Int(50, 60, 70))
    # Component C: 5-6
    G.add_segment(5, Vec3Int(10, 20, 30))
    G.add_segment(6, Vec3Int(40, 50, 60))
    G.add_affinity_edge(5, 6, affinity=1.0)
    attachment = AgglomerateAttachment.create(tmp_dir / "agg_view_multi", G)
    G2 = attachment.to_graph()
    assert_graphs_equal(G, G2)
