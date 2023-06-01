from http import HTTPStatus
from typing import Any, Dict

import httpx

from ...client import Client
from ...types import Response


def _get_kwargs(
    annotation_id: str,
    *,
    client: Client,
) -> Dict[str, Any]:
    url = "{}/api/zarrPrivateLinks/byAnnotation/{annotationId}".format(
        client.base_url, annotationId=annotation_id
    )

    headers: Dict[str, str] = client.get_headers()
    cookies: Dict[str, Any] = client.get_cookies()

    return {
        "method": "get",
        "url": url,
        "headers": headers,
        "cookies": cookies,
        "timeout": client.get_timeout(),
    }


def _build_response(*, response: httpx.Response) -> Response[Any]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=None,
    )


def sync_detailed(
    annotation_id: str,
    *,
    client: Client,
) -> Response[Any]:
    """List all existing private zarr links for a user for a given annotation

    Args:
        annotation_id (str):

    Returns:
        Response[Any]
    """

    kwargs = _get_kwargs(
        annotation_id=annotation_id,
        client=client,
    )

    response = httpx.request(
        verify=client.verify_ssl,
        **kwargs,
    )

    return _build_response(response=response)


async def asyncio_detailed(
    annotation_id: str,
    *,
    client: Client,
) -> Response[Any]:
    """List all existing private zarr links for a user for a given annotation

    Args:
        annotation_id (str):

    Returns:
        Response[Any]
    """

    kwargs = _get_kwargs(
        annotation_id=annotation_id,
        client=client,
    )

    async with httpx.AsyncClient(verify=client.verify_ssl) as _client:
        response = await _client.request(**kwargs)

    return _build_response(response=response)
