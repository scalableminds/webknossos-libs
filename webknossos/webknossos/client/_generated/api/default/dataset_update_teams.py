from http import HTTPStatus
from typing import Any, Dict, List, Optional

import httpx

from ... import errors
from ...client import Client
from ...types import Response


def _get_kwargs(
    organization_name: str,
    data_set_name: str,
    *,
    client: Client,
    json_body: List[str],
) -> Dict[str, Any]:
    url = "{}/api/datasets/{organizationName}/{dataSetName}/teams".format(
        client.base_url, organizationName=organization_name, dataSetName=data_set_name
    )

    headers: Dict[str, str] = client.get_headers()
    cookies: Dict[str, Any] = client.get_cookies()

    json_json_body = json_body

    return {
        "method": "patch",
        "url": url,
        "headers": headers,
        "cookies": cookies,
        "timeout": client.get_timeout(),
        "follow_redirects": client.follow_redirects,
        "json": json_json_body,
    }


def _parse_response(*, client: Client, response: httpx.Response) -> Optional[Any]:
    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(*, client: Client, response: httpx.Response) -> Response[Any]:
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
    json_body: List[str],
) -> Response[Any]:
    r"""\"Update teams of a dataset
    Expects:
     - As JSON object body:
       List of team strings.
     - As GET parameters:
      - organizationName (string): url-safe name of the organization owning the dataset
      - dataSetName (string): name of the dataset

    Args:
        organization_name (str):
        data_set_name (str):
        json_body (List[str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any]
    """

    kwargs = _get_kwargs(
        organization_name=organization_name,
        data_set_name=data_set_name,
        client=client,
        json_body=json_body,
    )

    response = httpx.request(
        verify=client.verify_ssl,
        **kwargs,
    )

    return _build_response(client=client, response=response)


async def asyncio_detailed(
    organization_name: str,
    data_set_name: str,
    *,
    client: Client,
    json_body: List[str],
) -> Response[Any]:
    r"""\"Update teams of a dataset
    Expects:
     - As JSON object body:
       List of team strings.
     - As GET parameters:
      - organizationName (string): url-safe name of the organization owning the dataset
      - dataSetName (string): name of the dataset

    Args:
        organization_name (str):
        data_set_name (str):
        json_body (List[str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any]
    """

    kwargs = _get_kwargs(
        organization_name=organization_name,
        data_set_name=data_set_name,
        client=client,
        json_body=json_body,
    )

    async with httpx.AsyncClient(verify=client.verify_ssl) as _client:
        response = await _client.request(**kwargs)

    return _build_response(client=client, response=response)
