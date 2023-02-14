from http import HTTPStatus
from typing import Any, Dict, Union

import httpx

from ...client import Client
from ...models.dataset_update_json_body import DatasetUpdateJsonBody
from ...types import UNSET, Response, Unset


def _get_kwargs(
    organization_name: str,
    data_set_name: str,
    *,
    client: Client,
    json_body: DatasetUpdateJsonBody,
    skip_resolutions: Union[Unset, None, bool] = UNSET,
) -> Dict[str, Any]:
    url = "{}/api/datasets/{organizationName}/{dataSetName}".format(
        client.base_url, organizationName=organization_name, dataSetName=data_set_name
    )

    headers: Dict[str, str] = client.get_headers()
    cookies: Dict[str, Any] = client.get_cookies()

    params: Dict[str, Any] = {}
    params["skipResolutions"] = skip_resolutions

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    json_json_body = json_body.to_dict()

    return {
        "method": "patch",
        "url": url,
        "headers": headers,
        "cookies": cookies,
        "timeout": client.get_timeout(),
        "json": json_json_body,
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
    *,
    client: Client,
    json_body: DatasetUpdateJsonBody,
    skip_resolutions: Union[Unset, None, bool] = UNSET,
) -> Response[Any]:
    """Update information for a dataset.
    Expects:
     - As JSON object body with keys:
      - description (optional string)
      - displayName (optional string)
      - sortingKey (optional long)
      - isPublic (boolean)
      - tags (list of string)
      - folderId (optional string)
     - As GET parameters:
      - organizationName (string): url-safe name of the organization owning the dataset
      - dataSetName (string): name of the dataset

    Args:
        organization_name (str):
        data_set_name (str):
        skip_resolutions (Union[Unset, None, bool]):
        json_body (DatasetUpdateJsonBody):

    Returns:
        Response[Any]
    """

    kwargs = _get_kwargs(
        organization_name=organization_name,
        data_set_name=data_set_name,
        client=client,
        json_body=json_body,
        skip_resolutions=skip_resolutions,
    )

    response = httpx.request(
        verify=client.verify_ssl,
        **kwargs,
    )

    return _build_response(response=response)


async def asyncio_detailed(
    organization_name: str,
    data_set_name: str,
    *,
    client: Client,
    json_body: DatasetUpdateJsonBody,
    skip_resolutions: Union[Unset, None, bool] = UNSET,
) -> Response[Any]:
    """Update information for a dataset.
    Expects:
     - As JSON object body with keys:
      - description (optional string)
      - displayName (optional string)
      - sortingKey (optional long)
      - isPublic (boolean)
      - tags (list of string)
      - folderId (optional string)
     - As GET parameters:
      - organizationName (string): url-safe name of the organization owning the dataset
      - dataSetName (string): name of the dataset

    Args:
        organization_name (str):
        data_set_name (str):
        skip_resolutions (Union[Unset, None, bool]):
        json_body (DatasetUpdateJsonBody):

    Returns:
        Response[Any]
    """

    kwargs = _get_kwargs(
        organization_name=organization_name,
        data_set_name=data_set_name,
        client=client,
        json_body=json_body,
        skip_resolutions=skip_resolutions,
    )

    async with httpx.AsyncClient(verify=client.verify_ssl) as _client:
        response = await _client.request(**kwargs)

    return _build_response(response=response)
