from collections.abc import Iterator

from webknossos.client.api_client.models import (
    ApiAdHocMeshInfo,
    ApiPrecomputedMeshInfo,
)

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
