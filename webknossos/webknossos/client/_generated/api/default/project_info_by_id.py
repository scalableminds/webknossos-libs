from http import HTTPStatus
from typing import Any, Dict, Optional, Union, cast

import httpx

from ...client import Client
from ...models.project_info_by_id_response_200 import ProjectInfoByIdResponse200
from ...types import Response


def _get_kwargs(
    id: str,
    *,
    client: Client,
) -> Dict[str, Any]:
    url = "{}/api/projects/{id}".format(client.base_url, id=id)

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
) -> Optional[Union[Any, ProjectInfoByIdResponse200]]:
    if response.status_code == 200:
        response_200 = ProjectInfoByIdResponse200.from_dict(response.json())

        return response_200
    if response.status_code == 400:
        response_400 = cast(Any, None)
        return response_400
    return None


def _build_response(
    *, response: httpx.Response
) -> Response[Union[Any, ProjectInfoByIdResponse200]]:
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
) -> Response[Union[Any, ProjectInfoByIdResponse200]]:
    """Information about a project selected by id

    Args:
        id (str):

    Returns:
        Response[Union[Any, ProjectInfoByIdResponse200]]
    """

    kwargs = _get_kwargs(
        id=id,
        client=client,
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
) -> Optional[Union[Any, ProjectInfoByIdResponse200]]:
    """Information about a project selected by id

    Args:
        id (str):

    Returns:
        Response[Union[Any, ProjectInfoByIdResponse200]]
    """

    return sync_detailed(
        id=id,
        client=client,
    ).parsed


async def asyncio_detailed(
    id: str,
    *,
    client: Client,
) -> Response[Union[Any, ProjectInfoByIdResponse200]]:
    """Information about a project selected by id

    Args:
        id (str):

    Returns:
        Response[Union[Any, ProjectInfoByIdResponse200]]
    """

    kwargs = _get_kwargs(
        id=id,
        client=client,
    )

    async with httpx.AsyncClient(verify=client.verify_ssl) as _client:
        response = await _client.request(**kwargs)

    return _build_response(response=response)


async def asyncio(
    id: str,
    *,
    client: Client,
) -> Optional[Union[Any, ProjectInfoByIdResponse200]]:
    """Information about a project selected by id

    Args:
        id (str):

    Returns:
        Response[Union[Any, ProjectInfoByIdResponse200]]
    """

    return (
        await asyncio_detailed(
            id=id,
            client=client,
        )
    ).parsed
