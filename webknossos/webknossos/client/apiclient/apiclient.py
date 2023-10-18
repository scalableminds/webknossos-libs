import logging
from typing import Dict, List, Optional, Type, TypeVar

import attr
import cattrs
import httpx

from webknossos.client.apiclient.models import ApiDataset, ApiShortLink

logger = logging.getLogger(__name__)

T = TypeVar("T")

Query = Dict[str, Optional[str]]


@attr.s(auto_attribs=True)
class ApiClient:
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

    def short_link_by_key(self, key: str) -> ApiShortLink:
        uri = f"{self._api_uri}/shortLinks/byKey/{key}"
        return self._get_json(uri, ApiShortLink)

    def dataset_info(
        self, organization_name, dataset_name, sharing_token: Optional[str]
    ) -> ApiDataset:
        uri = f"{self._api_uri}/datasets/{organization_name}/{dataset_name}"
        return self._get_json(uri, ApiDataset, query={"sharing_token": sharing_token})

    def dataset_list(
        self, is_active: Optional[bool], organization_name: Optional[str]
    ) -> List[ApiDataset]:
        uri = f"{self._api_uri}/datasets"
        return self._get_json(
            uri,
            List[ApiDataset],
            query={"isActive": is_active, "organizationName": organization_name},
        )

    def dataset_update_teams(
        self, organization_name: str, dataset_name: str, team_ids: List[str]
    ) -> None:
        uri = f"{self._api_uri}/datasets/{organization_name}/{dataset_name}/teams"
        self._patch_json(uri, team_ids)

    def dataset_update(
        self, organization_name: str, dataset_name: str, updated_dataset: ApiDataset
    ) -> None:
        uri = f"{self._api_uri}/datasets/{organization_name}/{dataset_name}"
        self._patch_json(uri, updated_dataset)

    # Private properties and methods

    @property
    def _api_uri(self) -> str:
        return f"{self.base_url}/api/v{self.webknossos_api_version}"

    def _get_json(
        self, uri: str, response_type: Type[T], query: Optional[Query] = None
    ) -> T:
        response = self._get(uri, query)
        return self._parse_json(response.json(), response_type)

    def _patch_json(self, uri, body_structured) -> None:
        body_json = cattrs.unstructure(body_structured)
        self._patch(uri, body_json)

    def _get(self, uri, query: Optional[Query] = None) -> httpx.Response:
        return self._request("GET", uri, query)

    def _patch(self, uri: str, body_json: Optional[str]) -> httpx.Response:
        return self._request("PATCH", uri, body_json=body_json)

    def _request(
        self,
        method: str,
        uri: str,
        query: Optional[Query] = None,
        body_json: Optional[str] = None,
    ) -> httpx.Response:
        response = httpx.request(
            method,
            uri,
            params=self._filter_query(query),
            json=body_json,
            headers=self.headers,
        )
        self._assert_good_response(uri, response)
        return response

    # Omit all entries where the value is None
    def _filter_query(self, query: Optional[Query]) -> Optional[Query]:
        if query is None:
            return None
        return {k: v for (k, v) in query.items() if v is not None}

    def _parse_json(self, response: httpx.Response, response_type: Type[T]) -> T:
        return cattrs.structure(
            response, response_type
        )  # TODO error handling? urlencode needed?

    def _assert_good_response(self, uri: str, response: httpx.Response) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.error(f"Response body: {str(response.content)[0:2000]}")
            logger.error(
                f"""An error occurred while performing a request to the uri {uri}.
If this is unexpected, please double-check your webknossos uri and credentials.
If the error persists, it might be caused by a version mismatch of the python client and the WEBKNOSSOS server API version.
See https://github.com/scalableminds/webknossos-libs/releases for current releases.

Response body: {str(response.content)[0:2000]}

"""
            )
            raise e
