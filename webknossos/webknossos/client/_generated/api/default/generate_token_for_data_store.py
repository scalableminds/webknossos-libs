from http import HTTPStatus
from typing import Any, Dict, Optional

import httpx

from ...client import Client
from ...models.generate_token_for_data_store_response_200 import (
    GenerateTokenForDataStoreResponse200,
)
from ...types import Response


def _get_kwargs(
    *,
    client: Client,
) -> Dict[str, Any]:
    url = "{}/api/userToken/generate".format(client.base_url)

    headers: Dict[str, str] = client.get_headers()
    cookies: Dict[str, Any] = client.get_cookies()

    return {
        "method": "post",
        "url": url,
        "headers": headers,
        "cookies": cookies,
        "timeout": client.get_timeout(),
    }


def _parse_response(
    *, response: httpx.Response
) -> Optional[GenerateTokenForDataStoreResponse200]:
    if response.status_code == 200:
        response_200 = GenerateTokenForDataStoreResponse200.from_dict(response.json())

        return response_200
    return None


def _build_response(
    *, response: httpx.Response
) -> Response[GenerateTokenForDataStoreResponse200]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(response=response),
    )


def sync_detailed(
    *,
    client: Client,
) -> Response[GenerateTokenForDataStoreResponse200]:
    """Generates a token that can be used for requests to a datastore. The token is valid for 1 day by
    default.

    Returns:
        Response[GenerateTokenForDataStoreResponse200]
    """

    kwargs = _get_kwargs(
        client=client,
    )

    response = httpx.request(
        verify=client.verify_ssl,
        **kwargs,
    )

    return _build_response(response=response)


def sync(
    *,
    client: Client,
) -> Optional[GenerateTokenForDataStoreResponse200]:
    """Generates a token that can be used for requests to a datastore. The token is valid for 1 day by
    default.

    Returns:
        Response[GenerateTokenForDataStoreResponse200]
    """

    return sync_detailed(
        client=client,
    ).parsed


async def asyncio_detailed(
    *,
    client: Client,
) -> Response[GenerateTokenForDataStoreResponse200]:
    """Generates a token that can be used for requests to a datastore. The token is valid for 1 day by
    default.

    Returns:
        Response[GenerateTokenForDataStoreResponse200]
    """

    kwargs = _get_kwargs(
        client=client,
    )

    async with httpx.AsyncClient(verify=client.verify_ssl) as _client:
        response = await _client.request(**kwargs)

    return _build_response(response=response)


async def asyncio(
    *,
    client: Client,
) -> Optional[GenerateTokenForDataStoreResponse200]:
    """Generates a token that can be used for requests to a datastore. The token is valid for 1 day by
    default.

    Returns:
        Response[GenerateTokenForDataStoreResponse200]
    """

    return (
        await asyncio_detailed(
            client=client,
        )
    ).parsed
