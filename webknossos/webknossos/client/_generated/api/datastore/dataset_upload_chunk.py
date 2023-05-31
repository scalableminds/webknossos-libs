from http import HTTPStatus
from typing import Any, Dict, Optional, Union

import httpx

from ... import errors
from ...client import Client
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    client: Client,
    token: Union[Unset, None, str] = UNSET,
) -> Dict[str, Any]:
    url = "{}/data/datasets".format(client.base_url)

    headers: Dict[str, str] = client.get_headers()
    cookies: Dict[str, Any] = client.get_cookies()

    params: Dict[str, Any] = {}
    params["token"] = token

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    return {
        "method": "post",
        "url": url,
        "headers": headers,
        "cookies": cookies,
        "timeout": client.get_timeout(),
        "follow_redirects": client.follow_redirects,
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
    token: Union[Unset, None, str] = UNSET,
) -> Response[Any]:
    r"""Upload a byte chunk for a new dataset
    Expects:
     - As file attachment: A raw byte chunk of the dataset
     - As form parameter:
      - name (string): dataset name
      - owningOrganization (string): owning organization name
      - resumableChunkNumber (int): chunk index
      - resumableChunkSize (int): chunk size in bytes
      - resumableTotalChunks (string): total chunk count of the upload
      - totalFileCount (string): total file count of the upload
      - resumableIdentifier (string): identifier of the resumable upload and file
    (\"{uploadId}/{filepath}\")
     - As GET parameter:
      - token (string): datastore token identifying the uploading user

    Args:
        token (Union[Unset, None, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any]
    """

    kwargs = _get_kwargs(
        client=client,
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
    token: Union[Unset, None, str] = UNSET,
) -> Response[Any]:
    r"""Upload a byte chunk for a new dataset
    Expects:
     - As file attachment: A raw byte chunk of the dataset
     - As form parameter:
      - name (string): dataset name
      - owningOrganization (string): owning organization name
      - resumableChunkNumber (int): chunk index
      - resumableChunkSize (int): chunk size in bytes
      - resumableTotalChunks (string): total chunk count of the upload
      - totalFileCount (string): total file count of the upload
      - resumableIdentifier (string): identifier of the resumable upload and file
    (\"{uploadId}/{filepath}\")
     - As GET parameter:
      - token (string): datastore token identifying the uploading user

    Args:
        token (Union[Unset, None, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any]
    """

    kwargs = _get_kwargs(
        client=client,
        token=token,
    )

    async with httpx.AsyncClient(verify=client.verify_ssl) as _client:
        response = await _client.request(**kwargs)

    return _build_response(client=client, response=response)
