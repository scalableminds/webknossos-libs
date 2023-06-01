from http import HTTPStatus
from typing import Any, Dict, Optional

import httpx

from ...client import Client
from ...models.short_link_by_key_response_200 import ShortLinkByKeyResponse200
from ...types import Response


def _get_kwargs(
    key: str,
    *,
    client: Client,
) -> Dict[str, Any]:
    url = "{}/api/shortLinks/byKey/{key}".format(client.base_url, key=key)

    headers: Dict[str, str] = client.get_headers()
    cookies: Dict[str, Any] = client.get_cookies()

    return {
        "method": "get",
        "url": url,
        "headers": headers,
        "cookies": cookies,
        "timeout": client.get_timeout(),
    }


def _parse_response(*, response: httpx.Response) -> Optional[ShortLinkByKeyResponse200]:
    if response.status_code == 200:
        response_200 = ShortLinkByKeyResponse200.from_dict(response.json())

        return response_200
    return None


def _build_response(*, response: httpx.Response) -> Response[ShortLinkByKeyResponse200]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(response=response),
    )


def sync_detailed(
    key: str,
    *,
    client: Client,
) -> Response[ShortLinkByKeyResponse200]:
    """Information about a short link, including the original long link.

    Args:
        key (str):

    Returns:
        Response[ShortLinkByKeyResponse200]
    """

    kwargs = _get_kwargs(
        key=key,
        client=client,
    )

    response = httpx.request(
        verify=client.verify_ssl,
        **kwargs,
    )

    return _build_response(response=response)


def sync(
    key: str,
    *,
    client: Client,
) -> Optional[ShortLinkByKeyResponse200]:
    """Information about a short link, including the original long link.

    Args:
        key (str):

    Returns:
        Response[ShortLinkByKeyResponse200]
    """

    return sync_detailed(
        key=key,
        client=client,
    ).parsed


async def asyncio_detailed(
    key: str,
    *,
    client: Client,
) -> Response[ShortLinkByKeyResponse200]:
    """Information about a short link, including the original long link.

    Args:
        key (str):

    Returns:
        Response[ShortLinkByKeyResponse200]
    """

    kwargs = _get_kwargs(
        key=key,
        client=client,
    )

    async with httpx.AsyncClient(verify=client.verify_ssl) as _client:
        response = await _client.request(**kwargs)

    return _build_response(response=response)


async def asyncio(
    key: str,
    *,
    client: Client,
) -> Optional[ShortLinkByKeyResponse200]:
    """Information about a short link, including the original long link.

    Args:
        key (str):

    Returns:
        Response[ShortLinkByKeyResponse200]
    """

    return (
        await asyncio_detailed(
            key=key,
            client=client,
        )
    ).parsed
