from http import HTTPStatus
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
    mag: str,
    half_byte: Union[Unset, None, bool] = False,
    mapping_name: Union[Unset, None, str] = UNSET,
) -> Dict[str, Any]:
    url = "{}/data/datasets/{organizationName}/{dataSetName}/layers/{dataLayerName}/data".format(
        client.base_url,
        organizationName=organization_name,
        dataSetName=data_set_name,
        dataLayerName=data_layer_name,
    )

    headers: Dict[str, str] = client.get_headers()
    cookies: Dict[str, Any] = client.get_cookies()

    params: Dict[str, Any] = {}
    params["token"] = token

    params["x"] = x

    params["y"] = y

    params["z"] = z

    params["width"] = width

    params["height"] = height

    params["depth"] = depth

    params["mag"] = mag

    params["halfByte"] = half_byte

    params["mappingName"] = mapping_name

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
    mag: str,
    half_byte: Union[Unset, None, bool] = False,
    mapping_name: Union[Unset, None, str] = UNSET,
) -> Response[Any]:
    """Get raw binary data from a bounding box in a dataset layer

    Args:
        organization_name (str):
        data_set_name (str):
        data_layer_name (str):
        token (Union[Unset, None, str]):
        x (int):
        y (int):
        z (int):
        width (int):
        height (int):
        depth (int):
        mag (str):
        half_byte (Union[Unset, None, bool]):
        mapping_name (Union[Unset, None, str]):

    Returns:
        Response[Any]
    """

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
        mag=mag,
        half_byte=half_byte,
        mapping_name=mapping_name,
    )

    response = httpx.request(
        verify=client.verify_ssl,
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
    mag: str,
    half_byte: Union[Unset, None, bool] = False,
    mapping_name: Union[Unset, None, str] = UNSET,
) -> Response[Any]:
    """Get raw binary data from a bounding box in a dataset layer

    Args:
        organization_name (str):
        data_set_name (str):
        data_layer_name (str):
        token (Union[Unset, None, str]):
        x (int):
        y (int):
        z (int):
        width (int):
        height (int):
        depth (int):
        mag (str):
        half_byte (Union[Unset, None, bool]):
        mapping_name (Union[Unset, None, str]):

    Returns:
        Response[Any]
    """

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
        mag=mag,
        half_byte=half_byte,
        mapping_name=mapping_name,
    )

    async with httpx.AsyncClient(verify=client.verify_ssl) as _client:
        response = await _client.request(**kwargs)

    return _build_response(response=response)
