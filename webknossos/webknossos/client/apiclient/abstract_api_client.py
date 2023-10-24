import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Type, TypeVar, Union

import cattrs
import httpx
import humps

logger = logging.getLogger(__name__)

T = TypeVar("T")

Query = Dict[str, Optional[Union[str, int, float, bool]]]

LONG_TIMEOUT_SECONDS = 7200.0


class AbstractApiClient(ABC):
    def __init__(
        self, timeout_seconds: float, headers: Optional[Dict[str, str]] = None
    ):
        self.headers = headers
        self.timeout_seconds = timeout_seconds

    @property
    @abstractmethod
    def url_prefix(self) -> str:
        ...

    def _get_json(
        self, route: str, response_type: Type[T], query: Optional[Query] = None
    ) -> T:
        response = self._get(route, query)
        return self._parse_json(response, response_type)

    def _get_json_paginated(
        self,
        route: str,
        response_type: Type[T],
        limit: Optional[int],
        page_number: Optional[int],
        query: Optional[Query] = None,
    ) -> Tuple[T, int]:
        pagination_query: Query = {
            "limit": limit,
            "pageNumber": page_number,
            "includeTotalCount": True,
        }
        query_adapted = pagination_query.copy()
        if query is not None:
            query_adapted.update(query)
        response = self._get(route, query_adapted)
        return self._parse_json(
            response, response_type
        ), self._extract_total_count_header(response)

    def _patch_json(self, route: str, body_structured: Any) -> None:
        body_json = self._prepare_for_json(body_structured)
        self._patch(route, body_json)

    def _post_json(
        self,
        route: str,
        body_structured: Any,
        query: Optional[Query] = None,
        retry_count: int = 1,
        timeout_seconds: Optional[float] = None,
    ) -> None:
        body_json = self._prepare_for_json(body_structured)
        self._post(route, body_json, query, retry_count, timeout_seconds)

    def _get(
        self,
        route: str,
        query: Optional[Query] = None,
        timeout_seconds: Optional[float] = None,
    ) -> httpx.Response:
        return self._request("GET", route, query, timeout_seconds=timeout_seconds)

    def _patch(
        self,
        route: str,
        body_json: Optional[Any],
        query: Optional[Query] = None,
        timeout_seconds: Optional[float] = None,
    ) -> httpx.Response:
        return self._request(
            "PATCH",
            route,
            body_json=body_json,
            query=query,
            timeout_seconds=timeout_seconds,
        )

    def _post(
        self,
        route: str,
        body_json: Optional[Any],
        query: Optional[Query] = None,
        retry_count: int = 1,
        timeout_seconds: Optional[float] = None,
    ) -> httpx.Response:
        return self._request(
            "POST",
            route,
            body_json=body_json,
            query=query,
            retry_count=retry_count,
            timeout_seconds=timeout_seconds,
        )

    def _request(
        self,
        method: str,
        route: str,
        query: Optional[Query] = None,
        body_json: Optional[Any] = None,
        retry_count: int = 1,
        timeout_seconds: Optional[float] = None,
    ) -> httpx.Response:
        assert (
            retry_count > 0
        ), f"Cannot perform request with retry_count < 1, got {retry_count}"
        url = f"{self.url_prefix}{route}"
        response = None
        for _ in range(retry_count):
            response = httpx.request(
                method,
                url,
                params=self._filter_query(query),
                json=body_json,
                headers=self.headers,
                timeout=timeout_seconds or self.timeout_seconds,
            )
            if response.status_code == 200 or response.status_code == 400:
                # Stop retrying in case of success or bad request
                break
        assert (
            response is not None
        ), "Got no http response. Was retry_count less than one?"
        self._assert_good_response(url, response)
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

    def _extract_total_count_header(self, response: httpx.Response) -> int:
        total_count_str = response.headers.get("X-Total-Count")
        assert total_count_str is not None, "X-Total-Count header missing from response"
        return int(total_count_str)

    def _prepare_for_json(self, body_structured: Any) -> Any:
        return cattrs.unstructure(humps.camelize(body_structured))

    def _assert_good_response(self, url: str, response: httpx.Response) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.error(
                f"""An error occurred while performing a request to the URL {url}.
If this is unexpected, please double-check your webknossos URL and credentials.
If the error persists, it might be caused by a version mismatch of the python client and the WEBKNOSSOS server API version.
See https://github.com/scalableminds/webknossos-libs/releases for current releases.

Response body: {str(response.content)[0:2000]}

"""
            )
            raise e
