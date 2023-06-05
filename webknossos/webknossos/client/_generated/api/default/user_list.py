from http import HTTPStatus
from typing import Any, Dict, List, Optional, Union

import httpx

from ...client import Client
from ...models.user_list_response_200_item import UserListResponse200Item
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    client: Client,
    is_editable: Union[Unset, None, bool] = UNSET,
    is_team_manager_or_admin: Union[Unset, None, bool] = UNSET,
    is_admin: Union[Unset, None, bool] = UNSET,
) -> Dict[str, Any]:
    url = "{}/api/users".format(client.base_url)

    headers: Dict[str, str] = client.get_headers()
    cookies: Dict[str, Any] = client.get_cookies()

    params: Dict[str, Any] = {}
    params["isEditable"] = is_editable

    params["isTeamManagerOrAdmin"] = is_team_manager_or_admin

    params["isAdmin"] = is_admin

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
) -> Optional[List["UserListResponse200Item"]]:
    if response.status_code == 200:
        response_200 = []
        _response_200 = response.json()
        for response_200_item_data in _response_200:
            response_200_item = UserListResponse200Item.from_dict(
                response_200_item_data
            )

            response_200.append(response_200_item)

        return response_200
    return None


def _build_response(
    *, response: httpx.Response
) -> Response[List["UserListResponse200Item"]]:
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
    is_team_manager_or_admin: Union[Unset, None, bool] = UNSET,
    is_admin: Union[Unset, None, bool] = UNSET,
) -> Response[List["UserListResponse200Item"]]:
    """List all users the requesting user is allowed to see (themself and users of whom they are admin or
    team-manager).

    Args:
        is_editable (Union[Unset, None, bool]):
        is_team_manager_or_admin (Union[Unset, None, bool]):
        is_admin (Union[Unset, None, bool]):

    Returns:
        Response[List['UserListResponse200Item']]
    """

    kwargs = _get_kwargs(
        client=client,
        is_editable=is_editable,
        is_team_manager_or_admin=is_team_manager_or_admin,
        is_admin=is_admin,
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
    is_team_manager_or_admin: Union[Unset, None, bool] = UNSET,
    is_admin: Union[Unset, None, bool] = UNSET,
) -> Optional[List["UserListResponse200Item"]]:
    """List all users the requesting user is allowed to see (themself and users of whom they are admin or
    team-manager).

    Args:
        is_editable (Union[Unset, None, bool]):
        is_team_manager_or_admin (Union[Unset, None, bool]):
        is_admin (Union[Unset, None, bool]):

    Returns:
        Response[List['UserListResponse200Item']]
    """

    return sync_detailed(
        client=client,
        is_editable=is_editable,
        is_team_manager_or_admin=is_team_manager_or_admin,
        is_admin=is_admin,
    ).parsed


async def asyncio_detailed(
    *,
    client: Client,
    is_editable: Union[Unset, None, bool] = UNSET,
    is_team_manager_or_admin: Union[Unset, None, bool] = UNSET,
    is_admin: Union[Unset, None, bool] = UNSET,
) -> Response[List["UserListResponse200Item"]]:
    """List all users the requesting user is allowed to see (themself and users of whom they are admin or
    team-manager).

    Args:
        is_editable (Union[Unset, None, bool]):
        is_team_manager_or_admin (Union[Unset, None, bool]):
        is_admin (Union[Unset, None, bool]):

    Returns:
        Response[List['UserListResponse200Item']]
    """

    kwargs = _get_kwargs(
        client=client,
        is_editable=is_editable,
        is_team_manager_or_admin=is_team_manager_or_admin,
        is_admin=is_admin,
    )

    async with httpx.AsyncClient(verify=client.verify_ssl) as _client:
        response = await _client.request(**kwargs)

    return _build_response(response=response)


async def asyncio(
    *,
    client: Client,
    is_editable: Union[Unset, None, bool] = UNSET,
    is_team_manager_or_admin: Union[Unset, None, bool] = UNSET,
    is_admin: Union[Unset, None, bool] = UNSET,
) -> Optional[List["UserListResponse200Item"]]:
    """List all users the requesting user is allowed to see (themself and users of whom they are admin or
    team-manager).

    Args:
        is_editable (Union[Unset, None, bool]):
        is_team_manager_or_admin (Union[Unset, None, bool]):
        is_admin (Union[Unset, None, bool]):

    Returns:
        Response[List['UserListResponse200Item']]
    """

    return (
        await asyncio_detailed(
            client=client,
            is_editable=is_editable,
            is_team_manager_or_admin=is_team_manager_or_admin,
            is_admin=is_admin,
        )
    ).parsed
