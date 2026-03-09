"""Tests for MeshAttachment.create(), parse_segment(), and parse_all()."""

import shutil
import tempfile
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest
from upath import UPath

from webknossos.dataset import (
    MeshAttachment,
    MeshfileMetadata,
    MeshFragment,
    MeshLod,
    SegmentMesh,
)
from webknossos.dataset.layer.segmentation_layer.attachments._utils import (
    read_zarr3_array,
)


@pytest.fixture()
def tmp_dir() -> Generator[Path, None, None]:
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)


def make_segment(segment_id: int, n_lods: int, frags_per_lod: int) -> SegmentMesh:
    """Build a SegmentMesh with dummy Draco bytes for testing."""
    lods = []
    for lod_idx in range(n_lods):
        fragments = [
            MeshFragment(
                position=(frag_idx, frag_idx + 1, frag_idx + 2),
                data=bytes([segment_id % 256, lod_idx % 256, frag_idx % 256, 0xFF]),
            )
            for frag_idx in range(frags_per_lod)
        ]
        lods.append(
            MeshLod(
                scale=float(2**lod_idx),
                vertex_offset=(0.0, 0.0, 0.0),
                fragments=fragments,
            )
        )
    return SegmentMesh(
        segment_id=segment_id,
        chunk_shape=(128.0, 128.0, 128.0),
        grid_origin=(0.0, 0.0, 0.0),
        lods=lods,
    )


def make_metadata(n_lod: int = 1) -> MeshfileMetadata:
    return MeshfileMetadata(
        global_bounding_box=(0, 0, 0, 1024, 1024, 512),
        mag=(1, 1, 1),
        mapping_name="test_mapping",
        vertex_quantization_bits_per_lod=[10] * n_lod,
        transform=np.eye(3, 4, dtype=np.float32),
        lod_scale_multiplier=1.0,
        unit="nm",
    )


def test_create_arrays_shape(tmp_dir: Path) -> None:
    segments = [make_segment(1, 2, 3), make_segment(2, 2, 3)]
    metadata = make_metadata(n_lod=2)
    mesh_path = tmp_dir / "mesh_view"
    attachment = MeshAttachment.create(mesh_path, segments, metadata)

    import math

    n_buckets = math.ceil(len(segments) / 0.75)

    meshfile_path = UPath(mesh_path / "meshfile")
    bucket_offsets = read_zarr3_array(meshfile_path / "bucket_offsets")
    buckets = read_zarr3_array(meshfile_path / "buckets")
    neuroglancer = read_zarr3_array(meshfile_path / "neuroglancer")

    assert bucket_offsets.shape == (n_buckets + 1,)
    assert bucket_offsets.dtype == np.uint64
    assert buckets.shape == (2, 3)
    assert buckets.dtype == np.uint64
    assert neuroglancer.ndim == 1
    assert neuroglancer.dtype == np.uint8

    assert attachment.name == "mesh_view"


def test_roundtrip_single_segment(tmp_dir: Path) -> None:
    segment = make_segment(42, 1, 2)
    metadata = make_metadata(n_lod=1)
    mesh_path = tmp_dir / "mesh_view"
    attachment = MeshAttachment.create(mesh_path, [segment], metadata)

    result = attachment.parse_segment(42)

    assert result.segment_id == 42
    assert result.chunk_shape == pytest.approx((128.0, 128.0, 128.0))
    assert result.grid_origin == pytest.approx((0.0, 0.0, 0.0))
    assert len(result.lods) == 1

    lod = result.lods[0]
    assert lod.scale == pytest.approx(1.0)
    assert len(lod.fragments) == 2

    orig_lod = segment.lods[0]
    for i, (frag, orig_frag) in enumerate(zip(lod.fragments, orig_lod.fragments)):
        assert frag.position == orig_frag.position, f"Position mismatch at fragment {i}"
        assert frag.data == orig_frag.data, f"Data mismatch at fragment {i}"


def test_roundtrip_multiple_segments(tmp_dir: Path) -> None:
    segments = [make_segment(sid, 2, 3) for sid in [10, 20, 30, 40, 50]]
    metadata = make_metadata(n_lod=2)
    mesh_path = tmp_dir / "mesh_view"
    attachment = MeshAttachment.create(mesh_path, segments, metadata)

    # Test parse_segment for each
    for orig in segments:
        result = attachment.parse_segment(orig.segment_id)
        assert result.segment_id == orig.segment_id
        assert len(result.lods) == len(orig.lods)
        for lod_idx, (res_lod, orig_lod) in enumerate(zip(result.lods, orig.lods)):
            assert res_lod.scale == pytest.approx(orig_lod.scale)
            assert len(res_lod.fragments) == len(orig_lod.fragments)
            for frag_idx, (res_frag, orig_frag) in enumerate(
                zip(res_lod.fragments, orig_lod.fragments)
            ):
                assert res_frag.position == orig_frag.position, (
                    f"seg {orig.segment_id} lod {lod_idx} frag {frag_idx}: position"
                )
                assert res_frag.data == orig_frag.data, (
                    f"seg {orig.segment_id} lod {lod_idx} frag {frag_idx}: data"
                )

    # Test parse_all
    all_results = list(attachment.parse_all())
    assert len(all_results) == len(segments)
    result_by_id = {r.segment_id: r for r in all_results}
    for orig in segments:
        assert orig.segment_id in result_by_id
        result = result_by_id[orig.segment_id]
        for lod_idx, (res_lod, orig_lod) in enumerate(zip(result.lods, orig.lods)):
            for frag_idx, (res_frag, orig_frag) in enumerate(
                zip(res_lod.fragments, orig_lod.fragments)
            ):
                assert res_frag.data == orig_frag.data, (
                    f"parse_all: seg {orig.segment_id} lod {lod_idx} frag {frag_idx}"
                )


def test_parse_segment_not_found(tmp_dir: Path) -> None:
    segment = make_segment(1, 1, 1)
    metadata = make_metadata(n_lod=1)
    mesh_path = tmp_dir / "mesh_view"
    attachment = MeshAttachment.create(mesh_path, [segment], metadata)

    with pytest.raises(KeyError):
        attachment.parse_segment(999)


def test_metadata_roundtrip(tmp_dir: Path) -> None:
    import json

    transform = np.array(
        [[1.5, 0.0, 0.0, 10.0], [0.0, 1.5, 0.0, 20.0], [0.0, 0.0, 2.0, 30.0]],
        dtype=np.float32,
    )
    metadata = MeshfileMetadata(
        global_bounding_box=(100, 200, 300, 512, 512, 256),
        mag=(2, 2, 1),
        mapping_name="agglomerate_view_30",
        vertex_quantization_bits_per_lod=[10, 12],
        transform=transform,
        lod_scale_multiplier=2.5,
        unit="um",
    )
    segments = [make_segment(1, 2, 1)]
    mesh_path = tmp_dir / "mesh_view"
    MeshAttachment.create(mesh_path, segments, metadata)

    meta = json.loads((mesh_path / "meshfile" / "zarr.json").read_text())
    attrs = meta["attributes"]["voxelytics"]

    assert attrs["version"] == 9
    assert attrs["global_bounding_box"] == [100, 200, 300, 512, 512, 256]
    assert attrs["mag"] == [2, 2, 1]
    assert attrs["mapping_name"] == "agglomerate_view_30"
    assert attrs["vertex_quantization_bits_per_lod"] == [10, 12]
    assert attrs["lod_scale_multiplier"] == pytest.approx(2.5)
    assert attrs["unit"] == "um"
    assert attrs["hash_function"] == "murmurhash3_x64_128"
    assert attrs["mesh_format"] == "draco"
    assert attrs["n_lod"] == 2

    stored_transform = np.array(attrs["transform"], dtype=np.float32)
    np.testing.assert_array_almost_equal(stored_transform, transform)
