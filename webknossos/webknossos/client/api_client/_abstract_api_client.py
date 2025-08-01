import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any, TypeVar

import httpx

from ...dataset.defaults import SSL_CONTEXT
from ._serialization import custom_converter
from .errors import CannotHandleResponseError, UnexpectedStatusError

logger = logging.getLogger(__name__)

T = TypeVar("T")

Query = dict[str, str | int | float | bool | None]

LONG_TIMEOUT_SECONDS = 7200.0  # 2 hours


class AbstractApiClient(ABC):
    def __init__(
        self,
        timeout_seconds: float,
        headers: dict[str, str] | None = None,
        webknossos_api_version: int = 9,
    ):
        self.headers = headers
        self.timeout_seconds = timeout_seconds
        self.webknossos_api_version = webknossos_api_version

    @property
    @abstractmethod
    def url_prefix(self) -> str: ...

    def url_from_route(self, route: str) -> str:
        return f"{self.url_prefix}{route}"

    def _get_json(
        self,
        route: str,
        response_type: type[T],
        query: Query | None = None,
        retry_count: int = 0,
    ) -> T:
        response = self._get(route, query, retry_count=retry_count)
        return self._parse_json(response, response_type)

    def _get_json_paginated(
        self,
        route: str,
        response_type: type[T],
        limit: int | None,
        page_number: int | None,
        query: Query | None = None,
    ) -> tuple[T, int]:
        pagination_query: Query = {
            "limit": limit,
            "pageNumber": page_number,
            "includeTotalCount": True,
        }
        if query is not None:
            pagination_query.update(query)
        response = self._get(route, pagination_query)
        return self._parse_json(
            response, response_type
        ), self._extract_total_count_header(response)

    def _put_json(self, route: str, body_structured: Any) -> None:
        body_json = self._prepare_for_json(body_structured)
        self._put(route, body_json)

    def _put_json_with_json_response(
        self,
        route: str,
        body_structured: Any,
        response_type: type[T],
        query: Query | None = None,
        retry_count: int = 0,
        timeout_seconds: float | None = None,
    ) -> T:
        body_json = self._prepare_for_json(body_structured)
        response = self._put(
            route,
            query=query,
            retry_count=retry_count,
            timeout_seconds=timeout_seconds,
            body_json=body_json,
        )
        return self._parse_json(response, response_type)

    def _patch_json(self, route: str, body_structured: Any) -> None:
        body_json = self._prepare_for_json(body_structured)
        self._patch(route, body_json)

    def _post_json(
        self,
        route: str,
        body_structured: Any,
        query: Query | None = None,
        retry_count: int = 0,
        timeout_seconds: float | None = None,
    ) -> None:
        body_json = self._prepare_for_json(body_structured)
        self._post(
            route,
            body_json=body_json,
            query=query,
            retry_count=retry_count,
            timeout_seconds=timeout_seconds,
        )

    def _post_json_with_bytes_iterator_response(
        self,
        route: str,
        body_structured: Any,
        query: Query | None = None,
        retry_count: int = 0,
        timeout_seconds: float | None = None,
    ) -> Iterator[bytes]:
        body_json = self._prepare_for_json(body_structured)
        response = self._post(
            route,
            body_json=body_json,
            query=query,
            retry_count=retry_count,
            timeout_seconds=timeout_seconds,
        )
        yield from response.iter_bytes()

    def _get_file(
        self, route: str, query: Query | None = None, retry_count: int = 0
    ) -> tuple[bytes, str]:
        response = self._get(route, query, retry_count=retry_count)
        return response.content, self._parse_filename_from_header(response)

    def _post_with_json_response(self, route: str, response_type: type[T]) -> T:
        response = self._post(route)
        return self._parse_json(response, response_type)

    def _post_json_with_json_response(
        self,
        route: str,
        body_structured: Any,
        response_type: type[T],
        query: Query | None = None,
        retry_count: int = 0,
        timeout_seconds: float | None = None,
    ) -> T:
        body_json = self._prepare_for_json(body_structured)
        response = self._post(
            route,
            query=query,
            retry_count=retry_count,
            timeout_seconds=timeout_seconds,
            body_json=body_json,
        )
        return self._parse_json(response, response_type)

    def post_multipart_with_json_response(
        self,
        route: str,
        response_type: type[T],
        multipart_data: httpx._types.RequestData | None = None,
        files: httpx._types.RequestFiles | None = None,
    ) -> T:
        response = self._post(route, multipart_data=multipart_data, files=files)
        return self._parse_json(response, response_type)

    def _get(
        self,
        route: str,
        query: Query | None = None,
        retry_count: int = 0,
        timeout_seconds: float | None = None,
    ) -> httpx.Response:
        return self._request(
            "GET",
            route,
            query,
            retry_count=retry_count,
            timeout_seconds=timeout_seconds,
        )

    def _patch(
        self,
        route: str,
        body_json: Any | None,
        query: Query | None = None,
        timeout_seconds: float | None = None,
    ) -> httpx.Response:
        return self._request(
            "PATCH",
            route,
            body_json=body_json,
            query=query,
            timeout_seconds=timeout_seconds,
        )

    def _put(
        self,
        route: str,
        body_json: Any | None = None,
        query: Query | None = None,
        multipart_data: httpx._types.RequestData | None = None,
        files: httpx._types.RequestFiles | None = None,
        retry_count: int = 0,
        timeout_seconds: float | None = None,
    ) -> httpx.Response:
        return self._request(
            "PUT",
            route,
            body_json=body_json,
            multipart_data=multipart_data,
            files=files,
            query=query,
            retry_count=retry_count,
            timeout_seconds=timeout_seconds,
        )

    def _post(
        self,
        route: str,
        body_json: Any | None = None,
        query: Query | None = None,
        multipart_data: httpx._types.RequestData | None = None,
        files: httpx._types.RequestFiles | None = None,
        retry_count: int = 0,
        timeout_seconds: float | None = None,
    ) -> httpx.Response:
        return self._request(
            "POST",
            route,
            body_json=body_json,
            multipart_data=multipart_data,
            files=files,
            query=query,
            retry_count=retry_count,
            timeout_seconds=timeout_seconds,
        )

    def _delete(
        self,
        route: str,
    ) -> httpx.Response:
        return self._request(
            "DELETE",
            route,
        )

    def _request(
        self,
        method: str,
        route: str,
        query: Query | None = None,
        body_json: Any | None = None,
        multipart_data: httpx._types.RequestData | None = None,
        files: httpx._types.RequestFiles | None = None,
        retry_count: int = 0,
        timeout_seconds: float | None = None,
    ) -> httpx.Response:
        assert retry_count >= 0, (
            f"Cannot perform request with retry_count < 0, got {retry_count}"
        )
        url = self.url_from_route(route)
        response = None
        for _ in range(retry_count + 1):
            response = httpx.request(
                method,
                url,
                params=self._omit_none_values_in_query(query),
                json=body_json,
                data=multipart_data,
                files=files,
                headers=self.headers,
                timeout=timeout_seconds or self.timeout_seconds,
                verify=SSL_CONTEXT,
            )
            if response.status_code == 200 or response.status_code == 400:
                # Stop retrying in case of success or bad request
                break
        assert response is not None, "Got no http response object"
        self._assert_good_response(response)
        return response

    def _omit_none_values_in_query(self, query: Query | None) -> Query | None:
        if query is None:
            return None
        return {k: v for (k, v) in query.items() if v is not None}

    def _parse_json(self, response: httpx.Response, response_type: type[T]) -> T:
        try:
            return custom_converter.structure(response.json(), response_type)
        except Exception as e:
            raise CannotHandleResponseError(response) from e

    def _extract_total_count_header(self, response: httpx.Response) -> int:
        total_count_str = response.headers.get("X-Total-Count")
        assert total_count_str is not None, "X-Total-Count header missing from response"
        return int(total_count_str)

    def _parse_filename_from_header(self, response: httpx.Response) -> str:
        # Adapted from https://peps.python.org/pep-0594/#cgi
        from email.message import Message

        content_disposition_str = response.headers.get("content-disposition", "")

        m = Message()
        m["content-type"] = content_disposition_str
        return dict(m.get_params() or []).get("filename", "")

    def _prepare_for_json(self, body_structured: Any) -> Any:
        return custom_converter.unstructure(body_structured)

    def _assert_good_response(self, response: httpx.Response) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise UnexpectedStatusError(response) from e
