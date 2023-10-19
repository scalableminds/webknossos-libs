from typing import Dict, Optional

from webknossos.client.apiclient.models import ApiUploadInformation

from .abstract_api_client import AbstractApiClient


class DatastoreApiClient(AbstractApiClient):
    def __init__(
        self, base_url: str, timeout: float, headers: Optional[Dict[str, str]] = None
    ):
        super().__init__(base_url, timeout, headers)

    @property
    def _datastore_uri(self) -> str:
        return f"{self.base_url}/data"

    def dataset_finish_upload(
        self,
        upload_information: ApiUploadInformation,
        token: Optional[str],
        retry_count: int,
    ) -> None:
        uri = f"{self._datastore_uri}/datasets/finishUpload"
        return self._post_json(
            uri, upload_information, query={"token": token}, retry_count=retry_count
        )
