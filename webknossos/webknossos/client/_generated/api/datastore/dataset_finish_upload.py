from http import HTTPStatus
from typing import Any, Dict, Optional, Union

import httpx

from ... import errors
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
        "follow_redirects": client.follow_redirects,
        "json": json_json_body,
        "params": params,
    }


def _parse_response(*, client: Client, response: httpx.Response) -> Optional[Any]:
    if response.status_code == HTTPStatus.OK:
        return None
    if response.status_code == HTTPStatus.BAD_REQUEST:
        return None
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

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

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

    return _build_response(client=client, response=response)


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

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

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

    return _build_response(client=client, response=response)
