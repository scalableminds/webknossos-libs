from webknossos.client.api_client.models import (
    ApiDatasetAnnounceUpload,
    ApiDatasetManualUploadSuccess,
    ApiDatasetUploadInformation,
    ApiDatasetUploadSuccess,
    ApiReserveDatasetUploadInformation,
)

from ._abstract_api_client import LONG_TIMEOUT_SECONDS, AbstractApiClient, Query


class DatastoreApiClient(AbstractApiClient):
    # Client to use the HTTP API of WEBKNOSSOS datastore servers.
    # When adding a method here, use the utility methods from AbstractApiClient
    # and add more as needed.
    # Methods here are prefixed with the domain, e.g. dataset_finish_upload (not finish_dataset_upload)

    def __init__(
        self,
        datastore_base_url: str,
        timeout_seconds: float,
        headers: dict[str, str] | None = None,
    ):
        super().__init__(timeout_seconds, headers)
        self.datastore_base_url = datastore_base_url.rstrip("/")

    @property
    def url_prefix(self) -> str:
        return f"{self.datastore_base_url}/data/v{self.webknossos_api_version}"

    def dataset_finish_upload(
        self,
        upload_information: ApiDatasetUploadInformation,
        token: str | None,
        retry_count: int,
    ) -> str:
        route = "/datasets/finishUpload"
        json = self._post_json_with_json_response(
            route,
            upload_information,
            query={"token": token},
            retry_count=retry_count,
            timeout_seconds=LONG_TIMEOUT_SECONDS,
            response_type=ApiDatasetUploadSuccess,
        )
        return json.new_dataset_id

    def dataset_reserve_upload(
        self,
        reserve_upload_information: ApiReserveDatasetUploadInformation,
        token: str | None,
        retry_count: int,
    ) -> None:
        route = "/datasets/reserveUpload"
        self._post_json(
            route,
            reserve_upload_information,
            query={"token": token},
            retry_count=retry_count,
        )

    def dataset_trigger_reload(
        self,
        organization_id: str,
        dataset_name: str,
        token: str | None = None,
    ) -> None:
        route = f"/triggers/reload/{organization_id}/{dataset_name}"
        query: Query = {"token": token}
        self._post(route, query=query)

    def dataset_reserve_manual_upload(
        self,
        dataset_announce: ApiDatasetAnnounceUpload,
        token: str | None,
    ) -> ApiDatasetManualUploadSuccess:
        route = "/datasets/reserveManualUpload"
        query: Query = {"token": token}
        return self._post_json_with_json_response(
            route, dataset_announce, ApiDatasetManualUploadSuccess, query
        )

    def dataset_get_raw_data(
        self,
        organization_id: str,
        directory_name: str,
        data_layer_name: str,
        mag: str,
        token: str | None,
        x: int,
        y: int,
        z: int,
        width: int,
        height: int,
        depth: int,
    ) -> tuple[bytes, str]:
        route = f"/datasets/{organization_id}/{directory_name}/layers/{data_layer_name}/data"
        query: Query = {
            "mag": mag,
            "x": x,
            "y": y,
            "z": z,
            "width": width,
            "height": height,
            "depth": depth,
            "token": token,
        }
        response = self._get(route, query)
        return response.content, response.headers.get("MISSING-BUCKETS")
