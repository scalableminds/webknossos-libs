from http import HTTPStatus
from typing import Any, Dict, List, Optional, Union, cast

import httpx

from ...client import Client
from ...models.task_infos_by_project_id_response_200_item import (
    TaskInfosByProjectIdResponse200Item,
)
from ...types import UNSET, Response, Unset


def _get_kwargs(
    id: str,
    *,
    client: Client,
    limit: Union[Unset, None, int] = UNSET,
    page_number: Union[Unset, None, int] = UNSET,
    include_total_count: Union[Unset, None, bool] = UNSET,
) -> Dict[str, Any]:
    url = "{}/api/projects/{id}/tasks".format(client.base_url, id=id)

    headers: Dict[str, str] = client.get_headers()
    cookies: Dict[str, Any] = client.get_cookies()

    params: Dict[str, Any] = {}
    params["limit"] = limit

    params["pageNumber"] = page_number

    params["includeTotalCount"] = include_total_count

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
) -> Optional[Union[Any, List["TaskInfosByProjectIdResponse200Item"]]]:
    if response.status_code == 200:
        response_200 = []
        _response_200 = response.json()
        for response_200_item_data in _response_200:
            response_200_item = TaskInfosByProjectIdResponse200Item.from_dict(
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
) -> Response[Union[Any, List["TaskInfosByProjectIdResponse200Item"]]]:
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
    limit: Union[Unset, None, int] = UNSET,
    page_number: Union[Unset, None, int] = UNSET,
    include_total_count: Union[Unset, None, bool] = UNSET,
) -> Response[Union[Any, List["TaskInfosByProjectIdResponse200Item"]]]:
    """Information about the tasks of a project selected by project id

    Args:
        id (str):
        limit (Union[Unset, None, int]):
        page_number (Union[Unset, None, int]):
        include_total_count (Union[Unset, None, bool]):

    Returns:
        Response[Union[Any, List['TaskInfosByProjectIdResponse200Item']]]
    """

    kwargs = _get_kwargs(
        id=id,
        client=client,
        limit=limit,
        page_number=page_number,
        include_total_count=include_total_count,
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
    limit: Union[Unset, None, int] = UNSET,
    page_number: Union[Unset, None, int] = UNSET,
    include_total_count: Union[Unset, None, bool] = UNSET,
) -> Optional[Union[Any, List["TaskInfosByProjectIdResponse200Item"]]]:
    """Information about the tasks of a project selected by project id

    Args:
        id (str):
        limit (Union[Unset, None, int]):
        page_number (Union[Unset, None, int]):
        include_total_count (Union[Unset, None, bool]):

    Returns:
        Response[Union[Any, List['TaskInfosByProjectIdResponse200Item']]]
    """

    return sync_detailed(
        id=id,
        client=client,
        limit=limit,
        page_number=page_number,
        include_total_count=include_total_count,
    ).parsed


async def asyncio_detailed(
    id: str,
    *,
    client: Client,
    limit: Union[Unset, None, int] = UNSET,
    page_number: Union[Unset, None, int] = UNSET,
    include_total_count: Union[Unset, None, bool] = UNSET,
) -> Response[Union[Any, List["TaskInfosByProjectIdResponse200Item"]]]:
    """Information about the tasks of a project selected by project id

    Args:
        id (str):
        limit (Union[Unset, None, int]):
        page_number (Union[Unset, None, int]):
        include_total_count (Union[Unset, None, bool]):

    Returns:
        Response[Union[Any, List['TaskInfosByProjectIdResponse200Item']]]
    """

    kwargs = _get_kwargs(
        id=id,
        client=client,
        limit=limit,
        page_number=page_number,
        include_total_count=include_total_count,
    )

    async with httpx.AsyncClient(verify=client.verify_ssl) as _client:
        response = await _client.request(**kwargs)

    return _build_response(response=response)


async def asyncio(
    id: str,
    *,
    client: Client,
    limit: Union[Unset, None, int] = UNSET,
    page_number: Union[Unset, None, int] = UNSET,
    include_total_count: Union[Unset, None, bool] = UNSET,
) -> Optional[Union[Any, List["TaskInfosByProjectIdResponse200Item"]]]:
    """Information about the tasks of a project selected by project id

    Args:
        id (str):
        limit (Union[Unset, None, int]):
        page_number (Union[Unset, None, int]):
        include_total_count (Union[Unset, None, bool]):

    Returns:
        Response[Union[Any, List['TaskInfosByProjectIdResponse200Item']]]
    """

    return (
        await asyncio_detailed(
            id=id,
            client=client,
            limit=limit,
            page_number=page_number,
            include_total_count=include_total_count,
        )
    ).parsed
