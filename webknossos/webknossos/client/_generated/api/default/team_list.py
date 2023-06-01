from http import HTTPStatus
from typing import Any, Dict, List, Optional, Union, cast

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

    headers: Dict[str, str] = client.get_headers()
    cookies: Dict[str, Any] = client.get_cookies()

    params: Dict[str, Any] = {}
    params["isEditable"] = is_editable

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
) -> Optional[Union[Any, List["TeamListResponse200Item"]]]:
    if response.status_code == 200:
        response_200 = []
        _response_200 = response.json()
        for response_200_item_data in _response_200:
            response_200_item = TeamListResponse200Item.from_dict(
                response_200_item_data
            )

            response_200.append(response_200_item)

        return response_200
    if response.status_code == 400:
        response_400 = cast(Any, None)
        return response_400
    return None


def _build_response(
    *, response: httpx.Response
) -> Response[Union[Any, List["TeamListResponse200Item"]]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(response=response),
    )


def sync_detailed(
    *,
    client: Client,
    is_editable: Union[Unset, None, bool] = UNSET,
) -> Response[Union[Any, List["TeamListResponse200Item"]]]:
    """List all accessible teams.

    Args:
        is_editable (Union[Unset, None, bool]):

    Returns:
        Response[Union[Any, List['TeamListResponse200Item']]]
    """

    kwargs = _get_kwargs(
        client=client,
        is_editable=is_editable,
    )

    response = httpx.request(
        verify=client.verify_ssl,
        **kwargs,
    )

    return _build_response(response=response)


def sync(
    *,
    client: Client,
    is_editable: Union[Unset, None, bool] = UNSET,
) -> Optional[Union[Any, List["TeamListResponse200Item"]]]:
    """List all accessible teams.

    Args:
        is_editable (Union[Unset, None, bool]):

    Returns:
        Response[Union[Any, List['TeamListResponse200Item']]]
    """

    return sync_detailed(
        client=client,
        is_editable=is_editable,
    ).parsed


async def asyncio_detailed(
    *,
    client: Client,
    is_editable: Union[Unset, None, bool] = UNSET,
) -> Response[Union[Any, List["TeamListResponse200Item"]]]:
    """List all accessible teams.

    Args:
        is_editable (Union[Unset, None, bool]):

    Returns:
        Response[Union[Any, List['TeamListResponse200Item']]]
    """

    kwargs = _get_kwargs(
        client=client,
        is_editable=is_editable,
    )

    async with httpx.AsyncClient(verify=client.verify_ssl) as _client:
        response = await _client.request(**kwargs)

    return _build_response(response=response)


async def asyncio(
    *,
    client: Client,
    is_editable: Union[Unset, None, bool] = UNSET,
) -> Optional[Union[Any, List["TeamListResponse200Item"]]]:
    """List all accessible teams.

    Args:
        is_editable (Union[Unset, None, bool]):

    Returns:
        Response[Union[Any, List['TeamListResponse200Item']]]
    """

    return (
        await asyncio_detailed(
            client=client,
            is_editable=is_editable,
        )
    ).parsed
