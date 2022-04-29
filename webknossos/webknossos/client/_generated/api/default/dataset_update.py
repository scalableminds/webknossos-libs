from typing import Any, Dict

import httpx

from ...client import Client
from ...models.dataset_update_json_body import DatasetUpdateJsonBody
from ...types import Response


def _get_kwargs(
    organization_name: str,
    data_set_name: str,
    *,
    client: Client,
    json_body: DatasetUpdateJsonBody,
) -> Dict[str, Any]:
    url = "{}/api/datasets/{organizationName}/{dataSetName}".format(
        client.base_url, organizationName=organization_name, dataSetName=data_set_name
    )

    headers: Dict[str, Any] = client.get_headers()
    cookies: Dict[str, Any] = client.get_cookies()

    json_json_body = json_body.to_dict()

    return {
        "url": url,
        "headers": headers,
        "cookies": cookies,
        "timeout": client.get_timeout(),
        "json": json_json_body,
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
    *,
    client: Client,
    json_body: DatasetUpdateJsonBody,
) -> Response[Any]:
    kwargs = _get_kwargs(
        organization_name=organization_name,
        data_set_name=data_set_name,
        client=client,
        json_body=json_body,
    )

    response = httpx.patch(
        **kwargs,
    )

    return _build_response(response=response)


async def asyncio_detailed(
    organization_name: str,
    data_set_name: str,
    *,
    client: Client,
    json_body: DatasetUpdateJsonBody,
) -> Response[Any]:
    kwargs = _get_kwargs(
        organization_name=organization_name,
        data_set_name=data_set_name,
        client=client,
        json_body=json_body,
    )

    async with httpx.AsyncClient() as _client:
        response = await _client.patch(**kwargs)

    return _build_response(response=response)
