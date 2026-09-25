import io
import zipfile
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import numpy as np
import pytest
import tensorstore

import webknossos as wk
from webknossos.annotation import RemoteAnnotation
from webknossos.client.api_client import _abstract_api_client
from webknossos.client.api_client.models import ApiAnnotationLayer
from webknossos.client.api_client.tracingstore_api_client import TracingStoreApiClient
from webknossos.client.context import _WebknossosContext
from webknossos.proofreading.generated import agglomerate_graph_pb2, list_of_long_pb2

TRACINGSTORE_URL = "http://tracingstore.test"
TRACING_ID = "volume-tracing-id"
ANNOTATION_ID = "annotation-id"


def _write_zarr3_array(path: Path, data: np.ndarray, chunk_shape: list[int]) -> None:
    # Same format as the tracingstore's editedEdgesZip response
    array = tensorstore.open(
        {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(path)},
            "metadata": {
                "shape": list(data.shape),
                "data_type": "bool" if data.dtype == bool else "uint64",
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": chunk_shape},
                },
                "chunk_key_encoding": {
                    "name": "v2",
                    "configuration": {"separator": "."},
                },
                "codecs": [
                    {"name": "bytes", "configuration": {"endian": "big"}},
                    {
                        "name": "blosc",
                        "configuration": {
                            "cname": "zstd",
                            "clevel": 5,
                            "shuffle": "shuffle",
                            "typesize": data.dtype.itemsize,
                        },
                    },
                ],
            },
            "create": True,
        }
    ).result()
    if data.size > 0:
        array.write(data).result()


def _edited_edges_zip(
    tmp_path: Path, edges: np.ndarray, is_addition: np.ndarray
) -> bytes:
    zarr_dir = tmp_path / "edited_edges"
    # Small chunks to also cover reading multiple chunks
    _write_zarr3_array(zarr_dir / "edges", edges.astype(np.uint64), [2, 2])
    _write_zarr3_array(zarr_dir / "edgeIsAddition", is_addition.astype(bool), [2])
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zip_file:
        for file in zarr_dir.rglob("*"):
            if file.is_file():
                zip_file.write(file, file.relative_to(zarr_dir).as_posix())
    return buffer.getvalue()


def _agglomerate_graph_proto(agglomerate_id: int) -> bytes:
    segments = [agglomerate_id * 100 + 1, agglomerate_id * 100 + 2]
    graph = agglomerate_graph_pb2.AgglomerateGraph(
        segments=segments,
        edges=[
            agglomerate_graph_pb2.AgglomerateEdge(
                source=segments[0], target=segments[1]
            )
        ],
        positions=[
            agglomerate_graph_pb2.Vec3IntProto(x=1, y=2, z=3),
            agglomerate_graph_pb2.Vec3IntProto(x=4, y=5, z=6),
        ],
        affinities=[0.5],
    )
    return graph.SerializeToString()


class FakeTracingStore:
    def __init__(
        self,
        edited_edges_zip: bytes,
        agglomerate_id_by_segment: dict[int, int],
    ) -> None:
        self.edited_edges_zip = edited_edges_zip
        self.agglomerate_id_by_segment = agglomerate_id_by_segment
        self.requests: list[httpx.Request] = []

    def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        request = httpx.Request(
            method,
            url,
            params=kwargs.get("params"),
            content=kwargs.get("content"),
            headers=kwargs.get("headers"),
        )
        self.requests.append(request)
        prefix = f"{TRACINGSTORE_URL}/tracings/mapping/{TRACING_ID}"
        path = str(request.url.copy_with(query=None))

        if method == "GET" and path == f"{prefix}/editedEdgesZip":
            return httpx.Response(200, content=self.edited_edges_zip, request=request)
        if method == "POST" and path == f"{prefix}/agglomeratesForSegments":
            assert request.url.params["annotationId"] == ANNOTATION_ID
            assert request.headers["Content-Type"] == "application/x-protobuf"
            segment_ids = list_of_long_pb2.ListOfLong.FromString(request.content).items
            assert list(segment_ids) == sorted(set(segment_ids))
            response = list_of_long_pb2.ListOfLong(
                items=[self.agglomerate_id_by_segment[s] for s in segment_ids]
            )
            return httpx.Response(
                200, content=response.SerializeToString(), request=request
            )
        if method == "GET" and path.startswith(f"{prefix}/agglomerateGraph/"):
            agglomerate_id = int(path.rsplit("/", 1)[1])
            return httpx.Response(
                200, content=_agglomerate_graph_proto(agglomerate_id), request=request
            )
        return httpx.Response(404, request=request)

    def paths(self, method: str) -> list[str]:
        return [r.url.path for r in self.requests if r.method == method]


@pytest.fixture
def annotation(monkeypatch: pytest.MonkeyPatch) -> RemoteAnnotation:
    annotation = RemoteAnnotation(
        annotation_id=ANNOTATION_ID,
        organization_id="organization",
        skeleton=wk.Skeleton(voxel_size=(11.24, 11.24, 25), dataset_name="dataset"),
        owner_name="owner",
    )
    annotation_info = SimpleNamespace(
        annotation_layers=[
            ApiAnnotationLayer(tracing_id="skeleton-id", typ="Skeleton", name="s"),
            ApiAnnotationLayer(tracing_id=TRACING_ID, typ="Volume", name="v"),
        ]
    )
    monkeypatch.setattr(
        RemoteAnnotation, "_get_annotation_info", lambda _self: annotation_info
    )
    monkeypatch.setattr(
        _WebknossosContext,
        "get_tracingstore_api_client",
        lambda _self: TracingStoreApiClient(
            base_url=TRACINGSTORE_URL, timeout_seconds=10
        ),
    )
    return annotation


@pytest.fixture
def fake_tracingstore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[FakeTracingStore]:
    edges = np.array([[1, 2], [3, 4], [2, 5], [6, 7], [1, 3]], dtype=np.uint64)
    is_addition = np.array([True, True, False, True, True])
    # 1-4 were merged into agglomerate 10 and 5 was split off into 20.
    # 7 maps to 0 (background), which must not be fetched as an agglomerate.
    agglomerate_id_by_segment = {1: 10, 2: 10, 3: 10, 4: 10, 5: 20, 6: 30, 7: 0}
    fake = FakeTracingStore(
        _edited_edges_zip(tmp_path, edges, is_addition), agglomerate_id_by_segment
    )
    monkeypatch.setattr(_abstract_api_client.httpx, "request", fake.request)
    yield fake


@pytest.mark.usefixtures("fake_tracingstore")
def test_get_edited_edges(annotation: RemoteAnnotation) -> None:
    edges, is_addition = annotation.get_edited_edges()

    assert edges.dtype == np.uint64
    assert is_addition.dtype == bool
    np.testing.assert_array_equal(
        edges, np.array([[1, 2], [3, 4], [2, 5], [6, 7], [1, 3]])
    )
    np.testing.assert_array_equal(is_addition, [True, True, False, True, True])


def test_get_edited_edges_empty(
    annotation: RemoteAnnotation,
    fake_tracingstore: FakeTracingStore,
    tmp_path: Path,
) -> None:
    fake_tracingstore.edited_edges_zip = _edited_edges_zip(
        tmp_path / "empty",
        np.zeros((0, 2), dtype=np.uint64),
        np.zeros((0,), dtype=bool),
    )

    edges, is_addition = annotation.get_edited_edges()

    assert edges.shape == (0, 2)
    assert is_addition.shape == (0,)


@pytest.mark.usefixtures("fake_tracingstore")
def test_get_agglomerate_ids_for_segments(annotation: RemoteAnnotation) -> None:
    # Unsorted with duplicates, the request must be sorted and deduplicated
    result = annotation.get_agglomerate_ids_for_segments([5, 1, 3, 1, 7])

    assert result == {1: 10, 3: 10, 5: 20, 7: 0}


def test_get_proofread_agglomerate_graph_data(
    annotation: RemoteAnnotation, fake_tracingstore: FakeTracingStore
) -> None:
    graphs = annotation.get_proofread_agglomerate_graph_data()

    assert list(graphs.keys()) == [10, 20, 30]
    np.testing.assert_array_equal(graphs[20].segments, [2001, 2002])
    np.testing.assert_array_equal(graphs[20].edges, [[2001, 2002]])
    assert graphs[20].to_agglomerate_graph().number_of_edges() == 1

    # Each touched agglomerate is fetched exactly once, background (0) is skipped
    graph_paths = [
        p for p in fake_tracingstore.paths("GET") if "/agglomerateGraph/" in p
    ]
    assert sorted(graph_paths) == [
        f"/tracings/mapping/{TRACING_ID}/agglomerateGraph/{i}" for i in (10, 20, 30)
    ]


def test_get_proofread_agglomerate_graph_data_without_edits(
    annotation: RemoteAnnotation,
    fake_tracingstore: FakeTracingStore,
    tmp_path: Path,
) -> None:
    fake_tracingstore.edited_edges_zip = _edited_edges_zip(
        tmp_path / "empty",
        np.zeros((0, 2), dtype=np.uint64),
        np.zeros((0,), dtype=bool),
    )

    assert annotation.get_proofread_agglomerate_graph_data() == {}
    assert fake_tracingstore.paths("POST") == []


def test_list_of_long_roundtrip_for_large_uint64() -> None:
    # Segment ids are unsigned 64 bit on the server
    values = [0, 1, 2**32, 2**63, 2**64 - 1]
    message = list_of_long_pb2.ListOfLong(items=values)
    assert (
        list(list_of_long_pb2.ListOfLong.FromString(message.SerializeToString()).items)
        == values
    )
