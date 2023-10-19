import logging
from typing import Any, Dict, List, Optional, Type, TypeVar

import attr
import cattrs
import httpx
import humps

from webknossos.client.apiclient.models import (
    ApiDataset,
    ApiSharingToken,
    ApiShortLink,
    ApiUploadInformation,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

Query = Dict[str, Optional[str]]


@attr.s(auto_attribs=True)
class AbstractApiClient():
    """A class for keeping track of data related to the API

    Attributes:
        base_url: The base URL for the API, all requests are made to a relative path to this URL
        headers: A dictionary of headers to be sent with every request
        timeout: The maximum amount of a time in seconds a request can take. API functions will raise
            httpx.TimeoutException if this is exceeded.
        webknossos_api_version: The webknossos REST Api version to use
    """

    base_url: str
    headers: Dict[str, str] = attr.ib(factory=dict, kw_only=True)
    timeout: float = attr.ib(5.0, kw_only=True)
    webknossos_api_version: int = attr.ib(5, kw_only=True)


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

    # Private properties and methods

    @property
    def _api_uri(self) -> str:
        return f"{self.base_url}/api/v{self.webknossos_api_version}"

    @property
    def _datastore_uri(self) -> str:
        return f"{self.base_url}/data"

    def _get_json(
        self, uri: str, response_type: Type[T], query: Optional[Query] = None
    ) -> T:
        response = self._get(uri, query)
        return self._parse_json(response, response_type)

    def _patch_json(self, uri, body_structured: Any) -> None:
        body_json = self._prepare_for_json(body_structured)
        self._patch(uri, body_json)

    def _post_json(
        self,
        uri,
        body_structured: Any,
        query: Optional[Query] = None,
        retry_count: int = 1,
    ) -> None:
        body_json = self._prepare_for_json(body_structured)
        self._post(uri, body_json, query, retry_count)

    def _get(self, uri, query: Optional[Query] = None) -> httpx.Response:
        return self._request("GET", uri, query)

    def _patch(
        self, uri: str, body_json: Optional[Any], query: Optional[Query] = None
    ) -> httpx.Response:
        return self._request("PATCH", uri, body_json=body_json, query=query)

    def _post(
        self,
        uri: str,
        body_json: Optional[Any],
        query: Optional[Query] = None,
        retry_count: int = 1,
    ) -> httpx.Response:
        return self._request(
            "POST", uri, body_json=body_json, query=query, retry_count=retry_count
        )

    def _request(
        self,
        method: str,
        uri: str,
        query: Optional[Query] = None,
        body_json: Optional[Any] = None,
        retry_count: int = 1,
    ) -> httpx.Response:
        assert (
            retry_count > 0
        ), f"Cannot perform request with retry_count < 1, got {retry_count}"
        response = None
        for _ in range(retry_count):
            response = httpx.request(
                method,
                uri,
                params=self._filter_query(query),
                json=body_json,
                headers=self.headers,
            )
            if response.status_code == 200 or response.status_code == 400:
                # Stop retrying in case of success or bad request
                break
        self._assert_good_response(uri, response)
        return response

    # Omit all entries where the value is None
    def _filter_query(self, query: Optional[Query]) -> Optional[Query]:
        if query is None:
            return None
        return {k: v for (k, v) in query.items() if v is not None}

    def _parse_json(self, response: httpx.Response, response_type: Type[T]) -> T:
        return cattrs.structure(
            humps.decamelize(response.json()), response_type
        )  # TODO error handling? urlencode needed?

    def _prepare_for_json(self, body_structured: Any) -> Any:
        return cattrs.unstructure(humps.camelize(body_structured))

    def _assert_good_response(
        self, uri: str, response: Optional[httpx.Response]
    ) -> None:
        assert (
            response is not None
        ), "Got no http response. Was retry_count less than one?"
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.error(
                f"""An error occurred while performing a request to the uri {uri}.
If this is unexpected, please double-check your webknossos uri and credentials.
If the error persists, it might be caused by a version mismatch of the python client and the WEBKNOSSOS server API version.
See https://github.com/scalableminds/webknossos-libs/releases for current releases.

Response body: {str(response.content)[0:2000]}

"""
            )
            raise e