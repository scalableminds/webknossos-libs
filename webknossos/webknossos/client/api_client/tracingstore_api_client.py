from collections.abc import Iterable, Iterator
from typing import Any

from webknossos.client.api_client.models import (
    ApiAdHocMeshInfo,
    ApiPrecomputedMeshInfo,
)

from ...proofreading.agglomerate_graph_data import AgglomerateGraphData
from ...proofreading.generated import agglomerate_graph_pb2, list_of_long_pb2
from ._abstract_api_client import AbstractApiClient, Query


class TracingStoreApiClient(AbstractApiClient):
    # Client to use the HTTP API of WEBKNOSSOS Tracing Store servers.
    # When adding a method here, use the utility methods from AbstractApiClient
    # and add more as needed.

    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float,
        headers: dict[str, str] | None = None,
    ):
        super().__init__(timeout_seconds, headers)
        self.base_url = base_url

    @property
    def url_prefix(self) -> str:
        return f"{self.base_url}/tracings"

    def annotation_download_mesh(
        self,
        *,
        mesh: ApiPrecomputedMeshInfo | ApiAdHocMeshInfo,
        tracing_id: str,
        token: str | None,
    ) -> Iterator[bytes]:
        route = f"/volume/{tracing_id}/fullMesh.stl"
        query: Query = {"token": token}
        yield from self._post_json_with_bytes_iterator_response(
            route=route,
            body_structured=mesh,
            query=query,
        )

    def get_agglomerate_graph(
        self, tracing_id: str, agglomerate_id: int
    ) -> AgglomerateGraphData:
        route = f"/mapping/{tracing_id}/agglomerateGraph/{agglomerate_id}"
        agglomerate_graph_proto = self._get_parsed_protobuf(
            route, agglomerate_graph_pb2.AgglomerateGraph
        )
        agglomerate_graph = AgglomerateGraphData.from_proto(agglomerate_graph_proto)
        return agglomerate_graph

    def annotation_newest_version(self, annotation_id: str) -> int:
        route = f"/annotation/{annotation_id}/newestVersion"
        return int(self._get(route).json()["version"])

    def annotation_update_action_log(
        self, annotation_id: str, newest_version: int, oldest_version: int
    ) -> list[tuple[int, list[dict[str, Any]]]]:
        """Returns the update groups in the version range, newest first."""
        route = f"/annotation/{annotation_id}/updateActionLog"
        query: Query = {
            "newestVersion": newest_version,
            "oldestVersion": oldest_version,
        }
        update_groups = self._get(route, query=query).json()
        return sorted(
            ((group["version"], group["value"]) for group in update_groups),
            key=lambda group: group[0],
            reverse=True,
        )

    def get_agglomerate_ids_for_segments(
        self, tracing_id: str, annotation_id: str, segment_ids: Iterable[int]
    ) -> dict[int, int]:
        # The server returns one agglomerate id per requested segment id, ordered by
        # segment id (not by request order), so the ids are sorted and deduplicated.
        sorted_segment_ids = sorted({int(segment_id) for segment_id in segment_ids})
        route = f"/mapping/{tracing_id}/agglomeratesForSegments"
        response = self._post_protobuf_with_protobuf_response(
            route=route,
            body=list_of_long_pb2.ListOfLong(items=sorted_segment_ids),
            MessageType=list_of_long_pb2.ListOfLong,
            query={"annotationId": annotation_id},
        )
        agglomerate_ids = list(response.items)
        assert len(agglomerate_ids) == len(sorted_segment_ids), (
            f"Expected {len(sorted_segment_ids)} agglomerate ids, "
            f"got {len(agglomerate_ids)}"
        )
        return dict(zip(sorted_segment_ids, agglomerate_ids))
