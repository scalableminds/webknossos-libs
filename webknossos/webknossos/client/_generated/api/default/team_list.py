from typing import Any, Dict, List, Optional, Union

import httpx

from ...client import Client
from ...models.team_list_response_200_item import TeamListResponse200Item
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    client: Client,
    is_editable: Union[Unset, None, bool] = UNSET,
) -> Dict[str, Any]:
    url = "{}/api/teams".format(client.base_url)

    headers: Dict[str, Any] = client.get_headers()
    cookies: Dict[str, Any] = client.get_cookies()

    params: Dict[str, Any] = {
        "isEditable": is_editable,
    }
    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    return {
        "url": url,
        "headers": headers,
        "cookies": cookies,
        "timeout": client.get_timeout(),
        "params": params,
    }


def _parse_response(
    *, response: httpx.Response
) -> Optional[List[TeamListResponse200Item]]:
    if response.status_code == 200:
        response_200 = []
        _response_200 = response.json()
        for response_200_item_data in _response_200:
            response_200_item = TeamListResponse200Item.from_dict(
                response_200_item_data
            )

            response_200.append(response_200_item)

        return response_200
    return None


def _build_response(
    *, response: httpx.Response
) -> Response[List[TeamListResponse200Item]]:
    return Response(
        status_code=response.status_code,
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(response=response),
    )


def sync_detailed(
    *,
    client: Client,
    is_editable: Union[Unset, None, bool] = UNSET,
) -> Response[List[TeamListResponse200Item]]:
    kwargs = _get_kwargs(
        client=client,
        is_editable=is_editable,
    )

    response = httpx.get(
        **kwargs,
    )

    return _build_response(response=response)


def sync(
    *,
    client: Client,
    is_editable: Union[Unset, None, bool] = UNSET,
) -> Optional[List[TeamListResponse200Item]]:
    """ """

    return sync_detailed(
        client=client,
        is_editable=is_editable,
    ).parsed


async def asyncio_detailed(
    *,
    client: Client,
    is_editable: Union[Unset, None, bool] = UNSET,
) -> Response[List[TeamListResponse200Item]]:
    kwargs = _get_kwargs(
        client=client,
        is_editable=is_editable,
    )

    async with httpx.AsyncClient() as _client:
        response = await _client.get(**kwargs)

    return _build_response(response=response)


async def asyncio(
    *,
    client: Client,
    is_editable: Union[Unset, None, bool] = UNSET,
) -> Optional[List[TeamListResponse200Item]]:
    """ """

    return (
        await asyncio_detailed(
            client=client,
            is_editable=is_editable,
        )
    ).parsed
