import attr

from typing import Dict, TypeVar, Type
from webknossos.client.apiclient.models import ApiShortLink
import httpx
import cattrs
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")


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

    # Private properties and methods

    @property
    def _api_uri(self) -> str:
        return f"{self.base_url}/api/v{self.webknossos_api_version}"

    def _get_json(self, uri: str, response_type: Type[T]) -> T:
        response = self._get(uri)
        return self._parse_json(response.json(), response_type)

    def _get(self, uri) -> httpx.Response:
        response = httpx.get(uri, headers=self.headers)
        self._assert_good_response(uri, response)
        return response

    def _parse_json(self, response: httpx.Response, response_type: Type[T]) -> T:
        return cattrs.structure(response, response_type)

    def _assert_good_response(self, uri: str, response: httpx.Response) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.error(f"""An error occurred while performing a request to the uri {uri}.
If this is unexpected, please double-check your webknossos uri and credentials.
If the error persists, it might be caused by a version mismatch of the python client and the WEBKNOSSOS server API version.
See https://github.com/scalableminds/webknossos-libs/releases for current releases.""")
            raise e
