from typing import Dict, Optional

from webknossos.client.apiclient.models import (
    ApiReserveUploadInformation,
    ApiUploadInformation,
)

from .abstract_api_client import LONG_TIMEOUT_SECONDS, AbstractApiClient


class DatastoreApiClient(AbstractApiClient):
    def __init__(
        self,
        datastore_base_url: str,
        timeout_seconds: float,
        headers: Optional[Dict[str, str]] = None,
    ):
        super().__init__(timeout_seconds, headers)
        self.datastore_base_url = datastore_base_url

    @property
    def url_prefix(self) -> str:
        return f"{self.datastore_base_url}/data"

    def dataset_finish_upload(
        self,
        upload_information: ApiUploadInformation,
        token: Optional[str],
        retry_count: int,
    ) -> None:
        route = f"/datasets/finishUpload"
        return self._post_json(
            route,
            upload_information,
            query={"token": token},
            retry_count=retry_count,
            timeout_seconds=LONG_TIMEOUT_SECONDS,
        )

    def dataset_reserve_upload(
        self,
        reserve_upload_information: ApiReserveUploadInformation,
        token: Optional[str],
        retry_count: int,
    ) -> None:
        route = f"/datasets/reserveUpload"
        return self._post_json(
            route,
            reserve_upload_information,
            query={"token": token},
            retry_count=retry_count,
        )
