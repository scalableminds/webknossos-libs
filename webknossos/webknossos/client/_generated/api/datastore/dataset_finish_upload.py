from http import HTTPStatus
from typing import Any, Dict, Union

import httpx

from ...client import Client
from ...models.dataset_finish_upload_json_body import DatasetFinishUploadJsonBody
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    client: Client,
    json_body: DatasetFinishUploadJsonBody,
    token: Union[Unset, None, str] = UNSET,
) -> Dict[str, Any]:
    url = "{}/data/datasets/finishUpload".format(client.base_url)

    headers: Dict[str, str] = client.get_headers()
    cookies: Dict[str, Any] = client.get_cookies()

    params: Dict[str, Any] = {}
    params["token"] = token

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    json_json_body = json_body.to_dict()

    return {
        "method": "post",
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
    *,
    client: Client,
    json_body: DatasetFinishUploadJsonBody,
    token: Union[Unset, None, str] = UNSET,
) -> Response[Any]:
    """Finish dataset upload, call after all chunks have been uploaded via uploadChunk
    Expects:
     - As JSON object body with keys:
      - uploadId (string): upload id that was also used in chunk upload (this time without file paths)
      - organization (string): owning organization name
      - name (string): dataset name
      - needsConversion (boolean): mark as true for non-wkw datasets. They are stored differently and a
    conversion job can later be run.
     - As GET parameter:
      - token (string): datastore token identifying the uploading user

    Args:
        token (Union[Unset, None, str]):
        json_body (DatasetFinishUploadJsonBody):

    Returns:
        Response[Any]
    """

    kwargs = _get_kwargs(
        client=client,
        json_body=json_body,
        token=token,
    )

    response = httpx.request(
        verify=client.verify_ssl,
        **kwargs,
    )

    return _build_response(response=response)


async def asyncio_detailed(
    *,
    client: Client,
    json_body: DatasetFinishUploadJsonBody,
    token: Union[Unset, None, str] = UNSET,
) -> Response[Any]:
    """Finish dataset upload, call after all chunks have been uploaded via uploadChunk
    Expects:
     - As JSON object body with keys:
      - uploadId (string): upload id that was also used in chunk upload (this time without file paths)
      - organization (string): owning organization name
      - name (string): dataset name
      - needsConversion (boolean): mark as true for non-wkw datasets. They are stored differently and a
    conversion job can later be run.
     - As GET parameter:
      - token (string): datastore token identifying the uploading user

    Args:
        token (Union[Unset, None, str]):
        json_body (DatasetFinishUploadJsonBody):

    Returns:
        Response[Any]
    """

    kwargs = _get_kwargs(
        client=client,
        json_body=json_body,
        token=token,
    )

    async with httpx.AsyncClient(verify=client.verify_ssl) as _client:
        response = await _client.request(**kwargs)

    return _build_response(response=response)
