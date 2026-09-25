import json
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import httpx
import numpy as np
import pytest

import webknossos as wk
from webknossos.annotation import RemoteAnnotation
from webknossos.annotation import annotation as annotation_module
from webknossos.client.api_client import _abstract_api_client
from webknossos.client.api_client.models import ApiAnnotationLayer
from webknossos.client.api_client.tracingstore_api_client import TracingStoreApiClient
from webknossos.client.context import _WebknossosContext
from webknossos.proofreading.edited_edges import edited_edges_from_update_groups
from webknossos.proofreading.generated import agglomerate_graph_pb2, list_of_long_pb2

TRACINGSTORE_URL = "http://tracingstore.test"
TRACING_ID = "volume-tracing-id"
ANNOTATION_ID = "annotation-id"


def _merge(segment_id1: Any, segment_id2: Any, tracing_id: str = TRACING_ID) -> dict:
    return {
        "name": "mergeAgglomerate",
        "value": {
            "segmentId1": segment_id1,
            "segmentId2": segment_id2,
            "actionTracingId": tracing_id,
        },
    }


def _split(segment_id1: Any, segment_id2: Any) -> dict:
    return {
        "name": "splitAgglomerate",
        "value": {
            "segmentId1": segment_id1,
            "segmentId2": segment_id2,
            "actionTracingId": TRACING_ID,
        },
    }


def _revert(source_version: int) -> dict:
    return {"name": "revertToVersion", "value": {"sourceVersion": source_version}}


def _bigint(value: int) -> dict:
    return {"customJsonEncoding": "bigint", "value": str(value)}


# Update action log of a proofreading annotation, by version.
# Version 3 is undone by the revert in version 4.
UPDATE_GROUPS: dict[int, list[dict]] = {
    0: [{"name": "updateMetadataOfAnnotation", "value": {"description": "x"}}],
    1: [_merge(1, 2)],
    2: [
        _merge(3, 4),
        {"name": "updateSegment", "value": {"id": 3, "actionTracingId": TRACING_ID}},
    ],
    3: [_merge(8, 9)],
    4: [_revert(2)],
    5: [_split(2, 5)],
    6: [_merge(6, _bigint(7)), _merge(100, 200, tracing_id="other-tracing-id")],
    7: [_merge(1, 3)],
}
EXPECTED_EDGES = [[1, 2], [3, 4], [2, 5], [6, 7], [1, 3]]
EXPECTED_IS_ADDITION = [True, True, False, True, True]


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
        update_groups: dict[int, list[dict]],
        agglomerate_id_by_segment: dict[int, int],
    ) -> None:
        self.update_groups = update_groups
        self.agglomerate_id_by_segment = agglomerate_id_by_segment
        self.requests: list[httpx.Request] = []

    def _json(self, request: httpx.Request, body: Any) -> httpx.Response:
        return httpx.Response(200, content=json.dumps(body), request=request)

    def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        request = httpx.Request(
            method,
            url,
            params=kwargs.get("params"),
            content=kwargs.get("content"),
            headers=kwargs.get("headers"),
        )
        self.requests.append(request)
        path = request.url.path
        annotation_prefix = f"/tracings/annotation/{ANNOTATION_ID}"
        mapping_prefix = f"/tracings/mapping/{TRACING_ID}"

        if method == "GET" and path == f"{annotation_prefix}/newestVersion":
            return self._json(request, {"version": max(self.update_groups, default=0)})
        if method == "GET" and path == f"{annotation_prefix}/updateActionLog":
            newest = int(request.url.params["newestVersion"])
            oldest = int(request.url.params["oldestVersion"])
            return self._json(
                request,
                [
                    {"version": version, "value": self.update_groups[version]}
                    for version in sorted(self.update_groups, reverse=True)
                    if oldest <= version <= newest
                ],
            )
        if method == "POST" and path == f"{mapping_prefix}/agglomeratesForSegments":
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
        if method == "GET" and path.startswith(f"{mapping_prefix}/agglomerateGraph/"):
            agglomerate_id = int(path.rsplit("/", 1)[1])
            return httpx.Response(
                200, content=_agglomerate_graph_proto(agglomerate_id), request=request
            )
        return httpx.Response(404, request=request)

    def requests_to(self, route_suffix: str) -> list[httpx.Request]:
        return [r for r in self.requests if r.url.path.endswith(route_suffix)]


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
def fake_tracingstore(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeTracingStore]:
    # 1-4 were merged into agglomerate 10 and 5 was split off into 20.
    # 7 maps to 0 (background), which must not be fetched as an agglomerate.
    agglomerate_id_by_segment = {1: 10, 2: 10, 3: 10, 4: 10, 5: 20, 6: 30, 7: 0}
    fake = FakeTracingStore(UPDATE_GROUPS, agglomerate_id_by_segment)
    monkeypatch.setattr(_abstract_api_client.httpx, "request", fake.request)
    yield fake


@pytest.mark.usefixtures("fake_tracingstore")
def test_get_edited_edges(annotation: RemoteAnnotation) -> None:
    edges, is_addition = annotation.get_edited_edges()

    assert edges.dtype == np.uint64
    assert is_addition.dtype == bool
    np.testing.assert_array_equal(edges, EXPECTED_EDGES)
    np.testing.assert_array_equal(is_addition, EXPECTED_IS_ADDITION)


def test_get_edited_edges_pages_the_update_log(
    annotation: RemoteAnnotation,
    fake_tracingstore: FakeTracingStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(annotation_module, "_UPDATE_ACTION_LOG_PAGE_SIZE", 3)

    edges, is_addition = annotation.get_edited_edges()

    np.testing.assert_array_equal(edges, EXPECTED_EDGES)
    np.testing.assert_array_equal(is_addition, EXPECTED_IS_ADDITION)
    requested_ranges = [
        (int(r.url.params["newestVersion"]), int(r.url.params["oldestVersion"]))
        for r in fake_tracingstore.requests_to("/updateActionLog")
    ]
    assert requested_ranges == [(7, 5), (4, 2), (1, 0)]


def test_get_edited_edges_empty(
    annotation: RemoteAnnotation, fake_tracingstore: FakeTracingStore
) -> None:
    fake_tracingstore.update_groups = {}

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

    # Reverted segments are not looked up
    lookup_request = fake_tracingstore.requests_to("/agglomeratesForSegments")[0]
    looked_up = list_of_long_pb2.ListOfLong.FromString(lookup_request.content).items
    assert list(looked_up) == [1, 2, 3, 4, 5, 6, 7]

    # Each touched agglomerate is fetched exactly once, background (0) is skipped
    graph_paths = sorted(
        r.url.path
        for r in fake_tracingstore.requests
        if "/agglomerateGraph/" in r.url.path
    )
    assert graph_paths == [
        f"/tracings/mapping/{TRACING_ID}/agglomerateGraph/{i}" for i in (10, 20, 30)
    ]


def test_get_proofread_agglomerate_graph_data_without_edits(
    annotation: RemoteAnnotation, fake_tracingstore: FakeTracingStore
) -> None:
    fake_tracingstore.update_groups = {0: UPDATE_GROUPS[0]}

    assert annotation.get_proofread_agglomerate_graph_data() == {}
    assert fake_tracingstore.requests_to("/agglomeratesForSegments") == []


def test_edited_edges_revert_of_a_revert() -> None:
    update_groups = [
        (5, [_merge(3, 4)]),
        # Reverts the first revert, so version 2 is active again
        (4, [_revert(2)]),
        (3, [_revert(1)]),
        (2, [_merge(1, 2)]),
        (1, []),
    ]

    edges, _ = edited_edges_from_update_groups(update_groups, tracing_id=TRACING_ID)

    np.testing.assert_array_equal(edges, [[1, 2], [3, 4]])


def test_edited_edges_large_segment_ids() -> None:
    update_groups = [(1, [_split(_bigint(2**64 - 1), 2**63)])]

    edges, is_addition = edited_edges_from_update_groups(
        update_groups, tracing_id=TRACING_ID
    )

    assert edges.tolist() == [[2**64 - 1, 2**63]]
    assert is_addition.tolist() == [False]


def test_edited_edges_skips_position_only_actions() -> None:
    legacy_merge = {
        "name": "mergeAgglomerate",
        "value": {
            "segmentPosition1": [1, 2, 3],
            "segmentPosition2": [4, 5, 6],
            "actionTracingId": TRACING_ID,
        },
    }
    update_groups = [(2, [_merge(1, 2)]), (1, [legacy_merge])]

    with pytest.warns(UserWarning, match="Skipped 1 proofreading actions"):
        edges, _ = edited_edges_from_update_groups(update_groups, tracing_id=TRACING_ID)

    np.testing.assert_array_equal(edges, [[1, 2]])


def test_list_of_long_roundtrip_for_large_uint64() -> None:
    # Segment ids are unsigned 64 bit on the server
    values = [0, 1, 2**32, 2**63, 2**64 - 1]
    message = list_of_long_pb2.ListOfLong(items=values)
    assert (
        list(list_of_long_pb2.ListOfLong.FromString(message.SerializeToString()).items)
        == values
    )
