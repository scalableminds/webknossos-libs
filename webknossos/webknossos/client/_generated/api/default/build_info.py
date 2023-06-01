from http import HTTPStatus
from typing import Any, Dict, Optional, Union, cast

import httpx

from ...client import Client
from ...models.build_info_response_200 import BuildInfoResponse200
from ...types import Response


def _get_kwargs(
    *,
    client: Client,
) -> Dict[str, Any]:
    url = "{}/api/buildinfo".format(client.base_url)

    headers: Dict[str, str] = client.get_headers()
    cookies: Dict[str, Any] = client.get_cookies()

    return {
        "method": "get",
        "url": url,
        "headers": headers,
        "cookies": cookies,
        "timeout": client.get_timeout(),
    }


def _parse_response(
    *, response: httpx.Response
) -> Optional[Union[Any, BuildInfoResponse200]]:
    if response.status_code == 200:
        response_200 = BuildInfoResponse200.from_dict(response.json())

        return response_200
    if response.status_code == 400:
        response_400 = cast(Any, None)
        return response_400
    return None


def _build_response(
    *, response: httpx.Response
) -> Response[Union[Any, BuildInfoResponse200]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(response=response),
    )


def sync_detailed(
    *,
    client: Client,
) -> Response[Union[Any, BuildInfoResponse200]]:
    """Information about the version of WEBKNOSSOS

    Returns:
        Response[Union[Any, BuildInfoResponse200]]
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
) -> Optional[Union[Any, BuildInfoResponse200]]:
    """Information about the version of WEBKNOSSOS

    Returns:
        Response[Union[Any, BuildInfoResponse200]]
    """

    return sync_detailed(
        client=client,
    ).parsed


async def asyncio_detailed(
    *,
    client: Client,
) -> Response[Union[Any, BuildInfoResponse200]]:
    """Information about the version of WEBKNOSSOS

    Returns:
        Response[Union[Any, BuildInfoResponse200]]
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
) -> Optional[Union[Any, BuildInfoResponse200]]:
    """Information about the version of WEBKNOSSOS

    Returns:
        Response[Union[Any, BuildInfoResponse200]]
    """

    return (
        await asyncio_detailed(
            client=client,
        )
    ).parsed
