from typing import Any, Dict, Union

import httpx

from ...client import Client
from ...types import UNSET, Response, Unset


def _get_kwargs(
    typ: str,
    id: str,
    *,
    client: Client,
    skeleton_version: Union[Unset, None, int] = UNSET,
    volume_version: Union[Unset, None, int] = UNSET,
    skip_volume_data: Union[Unset, None, bool] = UNSET,
) -> Dict[str, Any]:
    url = "{}/api/annotations/{typ}/{id}/download".format(
        client.base_url, typ=typ, id=id
    )

    headers: Dict[str, Any] = client.get_headers()
    cookies: Dict[str, Any] = client.get_cookies()

    params: Dict[str, Any] = {
        "skeletonVersion": skeleton_version,
        "volumeVersion": volume_version,
        "skipVolumeData": skip_volume_data,
    }
    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    return {
        "url": url,
        "headers": headers,
        "cookies": cookies,
        "timeout": client.get_timeout(),
        "params": params,
    }


def _build_response(*, response: httpx.Response) -> Response[Any]:
    return Response(
        status_code=response.status_code,
        content=response.content,
        headers=response.headers,
        parsed=None,
    )


def sync_detailed(
    typ: str,
    id: str,
    *,
    client: Client,
    skeleton_version: Union[Unset, None, int] = UNSET,
    volume_version: Union[Unset, None, int] = UNSET,
    skip_volume_data: Union[Unset, None, bool] = UNSET,
) -> Response[Any]:
    kwargs = _get_kwargs(
        typ=typ,
        id=id,
        client=client,
        skeleton_version=skeleton_version,
        volume_version=volume_version,
        skip_volume_data=skip_volume_data,
    )

    response = httpx.get(
        **kwargs,
    )

    return _build_response(response=response)


async def asyncio_detailed(
    typ: str,
    id: str,
    *,
    client: Client,
    skeleton_version: Union[Unset, None, int] = UNSET,
    volume_version: Union[Unset, None, int] = UNSET,
    skip_volume_data: Union[Unset, None, bool] = UNSET,
) -> Response[Any]:
    kwargs = _get_kwargs(
        typ=typ,
        id=id,
        client=client,
        skeleton_version=skeleton_version,
        volume_version=volume_version,
        skip_volume_data=skip_volume_data,
    )

    async with httpx.AsyncClient() as _client:
        response = await _client.get(**kwargs)

    return _build_response(response=response)
