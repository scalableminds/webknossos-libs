from typing import Any, Dict, List, Optional, Union

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

    headers: Dict[str, Any] = client.get_headers()
    cookies: Dict[str, Any] = client.get_cookies()

    params: Dict[str, Any] = {
        "limit": limit,
        "pageNumber": page_number,
        "includeTotalCount": include_total_count,
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
) -> Optional[List[TaskInfosByProjectIdResponse200Item]]:
    if response.status_code == 200:
        response_200 = []
        _response_200 = response.json()
        for response_200_item_data in _response_200:
            response_200_item = TaskInfosByProjectIdResponse200Item.from_dict(
                response_200_item_data
            )

            response_200.append(response_200_item)

        return response_200
    return None


def _build_response(
    *, response: httpx.Response
) -> Response[List[TaskInfosByProjectIdResponse200Item]]:
    return Response(
        status_code=response.status_code,
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
) -> Response[List[TaskInfosByProjectIdResponse200Item]]:
    kwargs = _get_kwargs(
        id=id,
        client=client,
        limit=limit,
        page_number=page_number,
        include_total_count=include_total_count,
    )

    response = httpx.get(
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
) -> Optional[List[TaskInfosByProjectIdResponse200Item]]:
    """ """

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
) -> Response[List[TaskInfosByProjectIdResponse200Item]]:
    kwargs = _get_kwargs(
        id=id,
        client=client,
        limit=limit,
        page_number=page_number,
        include_total_count=include_total_count,
    )

    async with httpx.AsyncClient() as _client:
        response = await _client.get(**kwargs)

    return _build_response(response=response)


async def asyncio(
    id: str,
    *,
    client: Client,
    limit: Union[Unset, None, int] = UNSET,
    page_number: Union[Unset, None, int] = UNSET,
    include_total_count: Union[Unset, None, bool] = UNSET,
) -> Optional[List[TaskInfosByProjectIdResponse200Item]]:
    """ """

    return (
        await asyncio_detailed(
            id=id,
            client=client,
            limit=limit,
            page_number=page_number,
            include_total_count=include_total_count,
        )
    ).parsed
