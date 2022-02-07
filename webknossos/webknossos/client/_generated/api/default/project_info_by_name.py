from typing import Any, Dict, Optional

import httpx

from ...client import Client
from ...models.project_info_by_name_response_200 import ProjectInfoByNameResponse200
from ...types import Response


def _get_kwargs(
    name: str,
    *,
    client: Client,
) -> Dict[str, Any]:
    url = "{}/api/projects/byName/{name}".format(client.base_url, name=name)

    headers: Dict[str, Any] = client.get_headers()
    cookies: Dict[str, Any] = client.get_cookies()

    return {
        "url": url,
        "headers": headers,
        "cookies": cookies,
        "timeout": client.get_timeout(),
    }


def _parse_response(
    *, response: httpx.Response
) -> Optional[ProjectInfoByNameResponse200]:
    if response.status_code == 200:
        response_200 = ProjectInfoByNameResponse200.from_dict(response.json())

        return response_200
    return None


def _build_response(
    *, response: httpx.Response
) -> Response[ProjectInfoByNameResponse200]:
    return Response(
        status_code=response.status_code,
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(response=response),
    )


def sync_detailed(
    name: str,
    *,
    client: Client,
) -> Response[ProjectInfoByNameResponse200]:
    kwargs = _get_kwargs(
        name=name,
        client=client,
    )

    response = httpx.get(
        **kwargs,
    )

    return _build_response(response=response)


def sync(
    name: str,
    *,
    client: Client,
) -> Optional[ProjectInfoByNameResponse200]:
    """ """

    return sync_detailed(
        name=name,
        client=client,
    ).parsed


async def asyncio_detailed(
    name: str,
    *,
    client: Client,
) -> Response[ProjectInfoByNameResponse200]:
    kwargs = _get_kwargs(
        name=name,
        client=client,
    )

    async with httpx.AsyncClient() as _client:
        response = await _client.get(**kwargs)

    return _build_response(response=response)


async def asyncio(
    name: str,
    *,
    client: Client,
) -> Optional[ProjectInfoByNameResponse200]:
    """ """

    return (
        await asyncio_detailed(
            name=name,
            client=client,
        )
    ).parsed
