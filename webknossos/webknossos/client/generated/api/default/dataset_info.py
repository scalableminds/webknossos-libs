from typing import Any, Dict, Optional, Union

import httpx

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

    headers: Dict[str, Any] = client.get_headers()
    cookies: Dict[str, Any] = client.get_cookies()

    params: Dict[str, Any] = {
        "sharingToken": sharing_token,
    }
    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    return {
        "url": url,
        "headers": headers,
        "cookies": cookies,
        "timeout": client.get_timeout(),
        "params": params,
    }


def _parse_response(*, response: httpx.Response) -> Optional[DatasetInfoResponse200]:
    if response.status_code == 200:
        response_200 = DatasetInfoResponse200.from_dict(response.json())

        return response_200
    return None


def _build_response(*, response: httpx.Response) -> Response[DatasetInfoResponse200]:
    return Response(
        status_code=response.status_code,
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(response=response),
    )


def sync_detailed(
    organization_name: str,
    data_set_name: str,
    *,
    client: Client,
    sharing_token: Union[Unset, None, str] = UNSET,
) -> Response[DatasetInfoResponse200]:
    kwargs = _get_kwargs(
        organization_name=organization_name,
        data_set_name=data_set_name,
        client=client,
        sharing_token=sharing_token,
    )

    response = httpx.get(
        **kwargs,
    )

    return _build_response(response=response)


def sync(
    organization_name: str,
    data_set_name: str,
    *,
    client: Client,
    sharing_token: Union[Unset, None, str] = UNSET,
) -> Optional[DatasetInfoResponse200]:
    """ """

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
) -> Response[DatasetInfoResponse200]:
    kwargs = _get_kwargs(
        organization_name=organization_name,
        data_set_name=data_set_name,
        client=client,
        sharing_token=sharing_token,
    )

    async with httpx.AsyncClient() as _client:
        response = await _client.get(**kwargs)

    return _build_response(response=response)


async def asyncio(
    organization_name: str,
    data_set_name: str,
    *,
    client: Client,
    sharing_token: Union[Unset, None, str] = UNSET,
) -> Optional[DatasetInfoResponse200]:
    """ """

    return (
        await asyncio_detailed(
            organization_name=organization_name,
            data_set_name=data_set_name,
            client=client,
            sharing_token=sharing_token,
        )
    ).parsed
