from http import HTTPStatus
from typing import Any, Dict, List, Optional, Union, cast

import httpx

from ...client import Client
from ...models.annotation_infos_by_task_id_response_200_item import (
    AnnotationInfosByTaskIdResponse200Item,
)
from ...types import Response


def _get_kwargs(
    id: str,
    *,
    client: Client,
) -> Dict[str, Any]:
    url = "{}/api/tasks/{id}/annotations".format(client.base_url, id=id)

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
) -> Optional[Union[Any, List["AnnotationInfosByTaskIdResponse200Item"]]]:
    if response.status_code == 200:
        response_200 = []
        _response_200 = response.json()
        for response_200_item_data in _response_200:
            response_200_item = AnnotationInfosByTaskIdResponse200Item.from_dict(
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
) -> Response[Union[Any, List["AnnotationInfosByTaskIdResponse200Item"]]]:
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
) -> Response[Union[Any, List["AnnotationInfosByTaskIdResponse200Item"]]]:
    """Information about all annotations for a specific task

    Args:
        id (str):

    Returns:
        Response[Union[Any, List['AnnotationInfosByTaskIdResponse200Item']]]
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
) -> Optional[Union[Any, List["AnnotationInfosByTaskIdResponse200Item"]]]:
    """Information about all annotations for a specific task

    Args:
        id (str):

    Returns:
        Response[Union[Any, List['AnnotationInfosByTaskIdResponse200Item']]]
    """

    return sync_detailed(
        id=id,
        client=client,
    ).parsed


async def asyncio_detailed(
    id: str,
    *,
    client: Client,
) -> Response[Union[Any, List["AnnotationInfosByTaskIdResponse200Item"]]]:
    """Information about all annotations for a specific task

    Args:
        id (str):

    Returns:
        Response[Union[Any, List['AnnotationInfosByTaskIdResponse200Item']]]
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
) -> Optional[Union[Any, List["AnnotationInfosByTaskIdResponse200Item"]]]:
    """Information about all annotations for a specific task

    Args:
        id (str):

    Returns:
        Response[Union[Any, List['AnnotationInfosByTaskIdResponse200Item']]]
    """

    return (
        await asyncio_detailed(
            id=id,
            client=client,
        )
    ).parsed
