from typing import Any, Dict, Union

import httpx

from ...client import Client
from ...types import UNSET, Response, Unset


def _get_kwargs(
    organization_name: str,
    data_set_name: str,
    data_layer_name: str,
    *,
    client: Client,
    token: Union[Unset, None, str] = UNSET,
    x: int,
    y: int,
    z: int,
    width: int,
    height: int,
    depth: int,
    resolution: int,
    half_byte: Union[Unset, None, bool] = False,
) -> Dict[str, Any]:
    url = "{}/data/datasets/{organizationName}/{dataSetName}/layers/{dataLayerName}/data".format(
        client.base_url,
        organizationName=organization_name,
        dataSetName=data_set_name,
        dataLayerName=data_layer_name,
    )

    headers: Dict[str, Any] = client.get_headers()
    cookies: Dict[str, Any] = client.get_cookies()

    params: Dict[str, Any] = {
        "token": token,
        "x": x,
        "y": y,
        "z": z,
        "width": width,
        "height": height,
        "depth": depth,
        "resolution": resolution,
        "halfByte": half_byte,
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
    organization_name: str,
    data_set_name: str,
    data_layer_name: str,
    *,
    client: Client,
    token: Union[Unset, None, str] = UNSET,
    x: int,
    y: int,
    z: int,
    width: int,
    height: int,
    depth: int,
    resolution: int,
    half_byte: Union[Unset, None, bool] = False,
) -> Response[Any]:
    kwargs = _get_kwargs(
        organization_name=organization_name,
        data_set_name=data_set_name,
        data_layer_name=data_layer_name,
        client=client,
        token=token,
        x=x,
        y=y,
        z=z,
        width=width,
        height=height,
        depth=depth,
        resolution=resolution,
        half_byte=half_byte,
    )

    response = httpx.get(
        **kwargs,
    )

    return _build_response(response=response)


async def asyncio_detailed(
    organization_name: str,
    data_set_name: str,
    data_layer_name: str,
    *,
    client: Client,
    token: Union[Unset, None, str] = UNSET,
    x: int,
    y: int,
    z: int,
    width: int,
    height: int,
    depth: int,
    resolution: int,
    half_byte: Union[Unset, None, bool] = False,
) -> Response[Any]:
    kwargs = _get_kwargs(
        organization_name=organization_name,
        data_set_name=data_set_name,
        data_layer_name=data_layer_name,
        client=client,
        token=token,
        x=x,
        y=y,
        z=z,
        width=width,
        height=height,
        depth=depth,
        resolution=resolution,
        half_byte=half_byte,
    )

    async with httpx.AsyncClient() as _client:
        response = await _client.get(**kwargs)

    return _build_response(response=response)
