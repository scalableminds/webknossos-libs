from http import HTTPStatus
from typing import Any, Dict, Union

import httpx

from ...client import Client
from ...types import UNSET, Response, Unset


def _get_kwargs(
    id: str,
    *,
    client: Client,
    skeleton_version: Union[Unset, None, int] = UNSET,
    volume_version: Union[Unset, None, int] = UNSET,
    skip_volume_data: Union[Unset, None, bool] = UNSET,
) -> Dict[str, Any]:
    url = "{}/api/annotations/{id}/download".format(client.base_url, id=id)

    headers: Dict[str, str] = client.get_headers()
    cookies: Dict[str, Any] = client.get_cookies()

    params: Dict[str, Any] = {}
    params["skeletonVersion"] = skeleton_version

    params["volumeVersion"] = volume_version

    params["skipVolumeData"] = skip_volume_data

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    return {
        "method": "get",
        "url": url,
        "headers": headers,
        "cookies": cookies,
        "timeout": client.get_timeout(),
        "params": params,
    }


def _build_response(*, response: httpx.Response) -> Response[Any]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=None,
    )


def sync_detailed(
    id: str,
    *,
    client: Client,
    skeleton_version: Union[Unset, None, int] = UNSET,
    volume_version: Union[Unset, None, int] = UNSET,
    skip_volume_data: Union[Unset, None, bool] = UNSET,
) -> Response[Any]:
    """Download an annotation as NML/ZIP

    Args:
        id (str):
        skeleton_version (Union[Unset, None, int]):
        volume_version (Union[Unset, None, int]):
        skip_volume_data (Union[Unset, None, bool]):

    Returns:
        Response[Any]
    """

    kwargs = _get_kwargs(
        id=id,
        client=client,
        skeleton_version=skeleton_version,
        volume_version=volume_version,
        skip_volume_data=skip_volume_data,
    )

    response = httpx.request(
        verify=client.verify_ssl,
        **kwargs,
    )

    return _build_response(response=response)


async def asyncio_detailed(
    id: str,
    *,
    client: Client,
    skeleton_version: Union[Unset, None, int] = UNSET,
    volume_version: Union[Unset, None, int] = UNSET,
    skip_volume_data: Union[Unset, None, bool] = UNSET,
) -> Response[Any]:
    """Download an annotation as NML/ZIP

    Args:
        id (str):
        skeleton_version (Union[Unset, None, int]):
        volume_version (Union[Unset, None, int]):
        skip_volume_data (Union[Unset, None, bool]):

    Returns:
        Response[Any]
    """

    kwargs = _get_kwargs(
        id=id,
        client=client,
        skeleton_version=skeleton_version,
        volume_version=volume_version,
        skip_volume_data=skip_volume_data,
    )

    async with httpx.AsyncClient(verify=client.verify_ssl) as _client:
        response = await _client.request(**kwargs)

    return _build_response(response=response)
