from http import HTTPStatus
from typing import Any, Dict, Optional, Union, cast

import httpx

from ... import errors
from ...client import Client
from ...models.dataset_info_response_200 import DatasetInfoResponse200
from ...types import UNSET, Response, Unset


def _get_kwargs(
    organization_name: str,
    data_set_name: str,
    *,
    client: Client,
    sharing_token: Union[Unset, None, str] = UNSET,
) -> Dict[str, Any]:
    url = "{}/api/datasets/{organizationName}/{dataSetName}".format(
        client.base_url, organizationName=organization_name, dataSetName=data_set_name
    )

    headers: Dict[str, str] = client.get_headers()
    cookies: Dict[str, Any] = client.get_cookies()

    params: Dict[str, Any] = {}
    params["sharingToken"] = sharing_token

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    return {
        "method": "get",
        "url": url,
        "headers": headers,
        "cookies": cookies,
        "timeout": client.get_timeout(),
        "follow_redirects": client.follow_redirects,
        "params": params,
    }


def _parse_response(
    *, client: Client, response: httpx.Response
) -> Optional[Union[Any, DatasetInfoResponse200]]:
    if response.status_code == HTTPStatus.OK:
        response_200 = DatasetInfoResponse200.from_dict(response.json())

        return response_200
    if response.status_code == HTTPStatus.BAD_REQUEST:
        response_400 = cast(Any, None)
        return response_400
    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: Client, response: httpx.Response
) -> Response[Union[Any, DatasetInfoResponse200]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    organization_name: str,
    data_set_name: str,
    *,
    client: Client,
    sharing_token: Union[Unset, None, str] = UNSET,
) -> Response[Union[Any, DatasetInfoResponse200]]:
    """Get information about this dataset

    Args:
        organization_name (str):
        data_set_name (str):
        sharing_token (Union[Unset, None, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, DatasetInfoResponse200]]
    """

    kwargs = _get_kwargs(
        organization_name=organization_name,
        data_set_name=data_set_name,
        client=client,
        sharing_token=sharing_token,
    )

    response = httpx.request(
        verify=client.verify_ssl,
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    organization_name: str,
    data_set_name: str,
    *,
    client: Client,
    sharing_token: Union[Unset, None, str] = UNSET,
) -> Optional[Union[Any, DatasetInfoResponse200]]:
    """Get information about this dataset

    Args:
        organization_name (str):
        data_set_name (str):
        sharing_token (Union[Unset, None, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, DatasetInfoResponse200]
    """

    return sync_detailed(
        organization_name=organization_name,
        data_set_name=data_set_name,
        client=client,
        sharing_token=sharing_token,
    ).parsed


async def asyncio_detailed(
    organization_name: str,
    data_set_name: str,
    *,
    client: Client,
    sharing_token: Union[Unset, None, str] = UNSET,
) -> Response[Union[Any, DatasetInfoResponse200]]:
    """Get information about this dataset

    Args:
        organization_name (str):
        data_set_name (str):
        sharing_token (Union[Unset, None, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, DatasetInfoResponse200]]
    """

    kwargs = _get_kwargs(
        organization_name=organization_name,
        data_set_name=data_set_name,
        client=client,
        sharing_token=sharing_token,
    )

    async with httpx.AsyncClient(verify=client.verify_ssl) as _client:
        response = await _client.request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    organization_name: str,
    data_set_name: str,
    *,
    client: Client,
    sharing_token: Union[Unset, None, str] = UNSET,
) -> Optional[Union[Any, DatasetInfoResponse200]]:
    """Get information about this dataset

    Args:
        organization_name (str):
        data_set_name (str):
        sharing_token (Union[Unset, None, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, DatasetInfoResponse200]
    """

    return (
        await asyncio_detailed(
            organization_name=organization_name,
            data_set_name=data_set_name,
            client=client,
            sharing_token=sharing_token,
        )
    ).parsed
