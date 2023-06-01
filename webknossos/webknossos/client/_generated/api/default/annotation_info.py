from http import HTTPStatus
from typing import Any, Dict, Optional, Union, cast

import httpx

from ...client import Client
from ...models.annotation_info_response_200 import AnnotationInfoResponse200
from ...types import UNSET, Response


def _get_kwargs(
    id: str,
    *,
    client: Client,
    timestamp: int,
) -> Dict[str, Any]:
    url = "{}/api/annotations/{id}/info".format(client.base_url, id=id)

    headers: Dict[str, str] = client.get_headers()
    cookies: Dict[str, Any] = client.get_cookies()

    params: Dict[str, Any] = {}
    params["timestamp"] = timestamp

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    return {
        "method": "get",
        "url": url,
        "headers": headers,
        "cookies": cookies,
        "timeout": client.get_timeout(),
        "params": params,
    }


def _parse_response(
    *, response: httpx.Response
) -> Optional[Union[AnnotationInfoResponse200, Any]]:
    if response.status_code == 200:
        response_200 = AnnotationInfoResponse200.from_dict(response.json())

        return response_200
    if response.status_code == 400:
        response_400 = cast(Any, None)
        return response_400
    return None


def _build_response(
    *, response: httpx.Response
) -> Response[Union[AnnotationInfoResponse200, Any]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(response=response),
    )


def sync_detailed(
    id: str,
    *,
    client: Client,
    timestamp: int,
) -> Response[Union[AnnotationInfoResponse200, Any]]:
    """Information about an annotation

    Args:
        id (str):
        timestamp (int):

    Returns:
        Response[Union[AnnotationInfoResponse200, Any]]
    """

    kwargs = _get_kwargs(
        id=id,
        client=client,
        timestamp=timestamp,
    )

    response = httpx.request(
        verify=client.verify_ssl,
        **kwargs,
    )

    return _build_response(response=response)


def sync(
    id: str,
    *,
    client: Client,
    timestamp: int,
) -> Optional[Union[AnnotationInfoResponse200, Any]]:
    """Information about an annotation

    Args:
        id (str):
        timestamp (int):

    Returns:
        Response[Union[AnnotationInfoResponse200, Any]]
    """

    return sync_detailed(
        id=id,
        client=client,
        timestamp=timestamp,
    ).parsed


async def asyncio_detailed(
    id: str,
    *,
    client: Client,
    timestamp: int,
) -> Response[Union[AnnotationInfoResponse200, Any]]:
    """Information about an annotation

    Args:
        id (str):
        timestamp (int):

    Returns:
        Response[Union[AnnotationInfoResponse200, Any]]
    """

    kwargs = _get_kwargs(
        id=id,
        client=client,
        timestamp=timestamp,
    )

    async with httpx.AsyncClient(verify=client.verify_ssl) as _client:
        response = await _client.request(**kwargs)

    return _build_response(response=response)


async def asyncio(
    id: str,
    *,
    client: Client,
    timestamp: int,
) -> Optional[Union[AnnotationInfoResponse200, Any]]:
    """Information about an annotation

    Args:
        id (str):
        timestamp (int):

    Returns:
        Response[Union[AnnotationInfoResponse200, Any]]
    """

    return (
        await asyncio_detailed(
            id=id,
            client=client,
            timestamp=timestamp,
        )
    ).parsed
