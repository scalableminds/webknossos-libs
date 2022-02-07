from typing import Any, Dict, Union

import httpx

from ...client import Client
from ...models.dataset_cancel_upload_json_body import DatasetCancelUploadJsonBody
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    client: Client,
    json_body: DatasetCancelUploadJsonBody,
    token: Union[Unset, None, str] = UNSET,
) -> Dict[str, Any]:
    url = "{}/data/datasets/cancelUpload".format(client.base_url)

    headers: Dict[str, Any] = client.get_headers()
    cookies: Dict[str, Any] = client.get_cookies()

    params: Dict[str, Any] = {
        "token": token,
    }
    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    json_json_body = json_body.to_dict()

    return {
        "url": url,
        "headers": headers,
        "cookies": cookies,
        "timeout": client.get_timeout(),
        "json": json_json_body,
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
    *,
    client: Client,
    json_body: DatasetCancelUploadJsonBody,
    token: Union[Unset, None, str] = UNSET,
) -> Response[Any]:
    kwargs = _get_kwargs(
        client=client,
        json_body=json_body,
        token=token,
    )

    response = httpx.post(
        **kwargs,
    )

    return _build_response(response=response)


async def asyncio_detailed(
    *,
    client: Client,
    json_body: DatasetCancelUploadJsonBody,
    token: Union[Unset, None, str] = UNSET,
) -> Response[Any]:
    kwargs = _get_kwargs(
        client=client,
        json_body=json_body,
        token=token,
    )

    async with httpx.AsyncClient() as _client:
        response = await _client.post(**kwargs)

    return _build_response(response=response)
