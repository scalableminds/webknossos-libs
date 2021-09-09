from typing import Any, Dict, List, Optional

import httpx

from ...client import Client
from ...models.datastore_list_response_200_item import DatastoreListResponse200Item
from ...types import Response


def _get_kwargs(
    *,
    client: Client,
) -> Dict[str, Any]:
    url = "{}/api/datastores".format(client.base_url)

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
) -> Optional[List[DatastoreListResponse200Item]]:
    if response.status_code == 200:
        response_200 = []
        _response_200 = response.json()
        for response_200_item_data in _response_200:
            response_200_item = DatastoreListResponse200Item.from_dict(
                response_200_item_data
            )

            response_200.append(response_200_item)

        return response_200
    return None


def _build_response(
    *, response: httpx.Response
) -> Response[List[DatastoreListResponse200Item]]:
    return Response(
        status_code=response.status_code,
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(response=response),
    )


def sync_detailed(
    *,
    client: Client,
) -> Response[List[DatastoreListResponse200Item]]:
    kwargs = _get_kwargs(
        client=client,
    )

    response = httpx.get(
        **kwargs,
    )

    return _build_response(response=response)


def sync(
    *,
    client: Client,
) -> Optional[List[DatastoreListResponse200Item]]:
    """ """

    return sync_detailed(
        client=client,
    ).parsed


async def asyncio_detailed(
    *,
    client: Client,
) -> Response[List[DatastoreListResponse200Item]]:
    kwargs = _get_kwargs(
        client=client,
    )

    async with httpx.AsyncClient() as _client:
        response = await _client.get(**kwargs)

    return _build_response(response=response)


async def asyncio(
    *,
    client: Client,
) -> Optional[List[DatastoreListResponse200Item]]:
    """ """

    return (
        await asyncio_detailed(
            client=client,
        )
    ).parsed
