import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Type, TypeVar, Union, Callable, Mapping, cast

import cattrs
import httpx
from attrs import fields as attr_fields
from attrs import has as is_attr_class
from attrs import AttrsInstance

from ...utils import snake_to_camel_case

logger = logging.getLogger(__name__)

T = TypeVar("T")

Query = Dict[str, Optional[Union[str, int, float, bool]]]

LONG_TIMEOUT_SECONDS = 7200.0

converter = cattrs.Converter()


def attr_to_camel_case_structure(cl: Type[T]) -> Callable[[Mapping[str, Any], Any], T]:
    return cattrs.gen.make_dict_structure_fn(
        cl,
        converter,
        **{
            a.name: cattrs.gen.override(rename=snake_to_camel_case(a.name))
            for a in attr_fields(cast(type[AttrsInstance], cl))
        },
    )


def attr_to_camel_case_unstructure(cl: Type[T]) -> Callable[[T], Dict[str, Any]]:
    return cattrs.gen.make_dict_unstructure_fn(
        cl,
        converter,
        **{
            a.name: cattrs.gen.override(rename=snake_to_camel_case(a.name))
            for a in attr_fields(cast(type[AttrsInstance], cl))
        },
    )


converter.register_structure_hook_factory(
    lambda cl: is_attr_class(cl), attr_to_camel_case_structure
)
converter.register_unstructure_hook_factory(
    lambda cl: is_attr_class(cl), attr_to_camel_case_unstructure
)


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
        self._post(
            route,
            body_json=body_json,
            query=query,
            retry_count=retry_count,
            timeout_seconds=timeout_seconds,
        )

    def _get_file(self, route: str, query: Optional[Query] = None) -> Tuple[bytes, str]:
        response = self._get(route, query)
        return response.content, self._parse_filename_from_header(response)

    def post_multipart_with_json_response(
        self,
        route: str,
        response_type: Type[T],
        multipart_data: Optional[httpx._types.RequestData] = None,
        files: Optional[httpx._types.RequestFiles] = None,
    ) -> T:
        response = self._post(route, multipart_data=multipart_data, files=files)
        return self._parse_json(response, response_type)

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
        body_json: Optional[Any] = None,
        query: Optional[Query] = None,
        multipart_data: Optional[httpx._types.RequestData] = None,
        files: Optional[httpx._types.RequestFiles] = None,
        retry_count: int = 1,
        timeout_seconds: Optional[float] = None,
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

    def _request(
        self,
        method: str,
        route: str,
        query: Optional[Query] = None,
        body_json: Optional[Any] = None,
        multipart_data: Optional[httpx._types.RequestData] = None,
        files: Optional[httpx._types.RequestFiles] = None,
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
                data=multipart_data,
                files=files,
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
        print(f"structuring {response.json()}")
        return converter.structure(
            response.json(), response_type
        )  # TODO error handling? urlencode needed?

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
        return converter.unstructure(body_structured)

    def _assert_good_response(self, url: str, response: httpx.Response) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            # todo move this from logging to exception body
            logger.error(
                f"""An error occurred while performing a request to the URL {url}.
If this is unexpected, please double-check your webknossos URL and credentials.
If the error persists, it might be caused by a version mismatch of the python client and the WEBKNOSSOS server API version.
See https://github.com/scalableminds/webknossos-libs/releases for current releases.

Response body: {str(response.content)[0:2000]}

"""
            )
            raise e
