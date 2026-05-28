from collections.abc import Iterator

from webknossos.client.api_client.models import (
    ApiAdHocMeshInfo,
    ApiAttachmentUploadInfo,
    ApiDatasetUploadInfo,
    ApiDatasetUploadInformationV13,
    ApiDatasetUploadSuccess,
    ApiDatasetUploadSuccessV13,
    ApiMagUploadInfo,
    ApiPrecomputedMeshInfo,
    ApiReserveDatasetUploadInformationV13,
)

from ._abstract_api_client import LONG_TIMEOUT_SECONDS, AbstractApiClient, Query


class DatastoreApiClient(AbstractApiClient):
    # Client to use the HTTP API of WEBKNOSSOS datastore servers.
    # When adding a method here, use the utility methods from AbstractApiClient
    # and add more as needed.
    # Methods here are prefixed with the domain, e.g. dataset_finish_upload (not finish_dataset_upload)

    def __init__(
        self,
        *,
        datastore_base_url: str,
        timeout_seconds: float,
        headers: dict[str, str] | None = None,
    ):
        super().__init__(timeout_seconds, headers)
        self.datastore_base_url = datastore_base_url.rstrip("/")

    @property
    def url_prefix(self) -> str:
        return f"{self.datastore_base_url}/data/v{self.webknossos_api_version}"

    def dataset_upload_resumable_url(self) -> str:
        return f"{self.url_prefix}/datasets/upload/dataset"

    def dataset_upload_resumable_query(
        self, _organization_id: str, _dataset_name: str, total_file_count: int
    ) -> dict:
        return {"totalFileCount": total_file_count}

    def mag_reserve_upload(
        self, *, mag_upload_info: ApiMagUploadInfo, retry_count: int
    ) -> None:
        self._post_json(
            "/datasets/upload/mag/reserveUpload",
            mag_upload_info,
            retry_count=retry_count,
        )

    def mag_finish_upload(
        self,
        *,
        upload_id: str,
        retry_count: int,
    ) -> None:
        self._post(
            "/datasets/upload/mag/finishUpload",
            query={"uploadId": upload_id},
            retry_count=retry_count,
            timeout_seconds=LONG_TIMEOUT_SECONDS,
        )

    def attachment_reserve_upload(
        self, *, attachment_upload_info: ApiAttachmentUploadInfo, retry_count: int
    ) -> None:
        self._post_json(
            "/datasets/upload/attachment/reserveUpload",
            attachment_upload_info,
            retry_count=retry_count,
        )

    def attachment_finish_upload(
        self,
        *,
        upload_id: str,
        retry_count: int,
    ) -> None:
        self._post(
            "/datasets/upload/attachment/finishUpload",
            query={"uploadId": upload_id},
            retry_count=retry_count,
            timeout_seconds=LONG_TIMEOUT_SECONDS,
        )

    def dataset_finish_upload(
        self,
        *,
        upload_id: str,
        retry_count: int,
    ) -> str:
        route = "/datasets/upload/dataset/finishUpload"
        json = self._post_with_json_response(
            route,
            query={"uploadId": upload_id},
            retry_count=retry_count,
            timeout_seconds=LONG_TIMEOUT_SECONDS,
            response_type=ApiDatasetUploadSuccess,
        )
        return json.dataset_id

    def dataset_reserve_upload(
        self,
        *,
        dataset_upload_info: ApiDatasetUploadInfo,
        retry_count: int,
    ) -> None:
        route = "/datasets/upload/dataset/reserveUpload"
        self._post_json(
            route,
            dataset_upload_info,
            retry_count=retry_count,
        )

    def dataset_trigger_reload(
        self,
        *,
        organization_id: str,
        dataset_id: str,
    ) -> None:
        route = f"/triggers/reload/{organization_id}/{dataset_id}"
        self._post(route)

    def dataset_get_raw_data(
        self,
        *,
        dataset_id: str,
        data_layer_name: str,
        mag: str,
        sharing_token: str | None,
        x: int,
        y: int,
        z: int,
        width: int,
        height: int,
        depth: int,
    ) -> tuple[bytes, str]:
        route = f"/datasets/{dataset_id}/layers/{data_layer_name}/data"
        query: Query = {
            "mag": mag,
            "x": x,
            "y": y,
            "z": z,
            "width": width,
            "height": height,
            "depth": depth,
            "token": sharing_token,
        }
        response = self._get(route, query)
        return response.content, response.headers.get("MISSING-BUCKETS")

    def download_mesh(
        self,
        *,
        mesh_info: ApiPrecomputedMeshInfo | ApiAdHocMeshInfo,
        dataset_id: str,
        layer_name: str,
        sharing_token: str | None,
    ) -> Iterator[bytes]:
        route = f"/datasets/{dataset_id}/layers/{layer_name}/meshes/fullMesh.stl"
        query: Query = {"token": sharing_token}
        yield from self._post_json_with_bytes_iterator_response(
            route=route,
            body_structured=mesh_info,
            query=query,
        )


class DatastoreApiClientV13(DatastoreApiClient):
    def __init__(
        self,
        *,
        datastore_base_url: str,
        timeout_seconds: float,
        headers: dict[str, str] | None = None,
    ):
        super().__init__(
            datastore_base_url=datastore_base_url,
            timeout_seconds=timeout_seconds,
            headers=headers,
        )
        self.webknossos_api_version = 13

    def dataset_reserve_upload(
        self,
        *,
        dataset_upload_info: ApiDatasetUploadInfo,
        retry_count: int,
    ) -> None:
        v13_body = ApiReserveDatasetUploadInformationV13(
            upload_id=dataset_upload_info.resumable_upload_info.upload_id,
            name=dataset_upload_info.dataset_name,
            organization=dataset_upload_info.organization_id,
            total_file_count=dataset_upload_info.resumable_upload_info.total_file_count,
            total_file_size_in_bytes=dataset_upload_info.resumable_upload_info.total_file_size_in_bytes,
            initial_teams=dataset_upload_info.initial_team_ids,
            layers_to_link=dataset_upload_info.layers_to_link,
            folder_id=dataset_upload_info.folder_id,
        )
        self._post_json("/datasets/reserveUpload", v13_body, retry_count=retry_count)

    def dataset_finish_upload(
        self,
        *,
        upload_id: str,
        retry_count: int,
    ) -> str:
        json = self._post_json_with_json_response(
            "/datasets/finishUpload",
            ApiDatasetUploadInformationV13(upload_id=upload_id),
            retry_count=retry_count,
            timeout_seconds=LONG_TIMEOUT_SECONDS,
            response_type=ApiDatasetUploadSuccessV13,
        )
        return json.new_dataset_id

    def dataset_upload_resumable_url(self) -> str:
        return f"{self.datastore_base_url}/data/datasets"

    def dataset_upload_resumable_query(
        self, _organization_id: str, _dataset_name: str, total_file_count: int
    ) -> dict:
        return {
            "owningOrganization": _organization_id,
            "name": _dataset_name,
            "totalFileCount": total_file_count,
        }

    def mag_reserve_upload(
        self, *, mag_upload_info: ApiMagUploadInfo, retry_count: int
    ) -> None:
        raise NotImplementedError("mag upload requires API version 14+")

    def mag_finish_upload(self, *, upload_id: str, retry_count: int) -> None:
        raise NotImplementedError("mag upload requires API version 14+")

    def attachment_reserve_upload(
        self, *, attachment_upload_info: ApiAttachmentUploadInfo, retry_count: int
    ) -> None:
        raise NotImplementedError("attachment upload requires API version 14+")

    def attachment_finish_upload(self, *, upload_id: str, retry_count: int) -> None:
        raise NotImplementedError("attachment upload requires API version 14+")
