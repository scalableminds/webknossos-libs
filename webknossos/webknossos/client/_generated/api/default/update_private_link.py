from http import HTTPStatus
from typing import Any, Dict, Optional

import httpx

from ... import errors
from ...client import Client
from ...models.update_private_link_json_body import UpdatePrivateLinkJsonBody
from ...types import Response


def _get_kwargs(
    id: str,
    *,
    client: Client,
    json_body: UpdatePrivateLinkJsonBody,
) -> Dict[str, Any]:
    url = "{}/api/zarrPrivateLinks/{id}".format(client.base_url, id=id)

    headers: Dict[str, str] = client.get_headers()
    cookies: Dict[str, Any] = client.get_cookies()

    json_json_body = json_body.to_dict()

    return {
        "method": "put",
        "url": url,
        "headers": headers,
        "cookies": cookies,
        "timeout": client.get_timeout(),
        "follow_redirects": client.follow_redirects,
        "json": json_json_body,
    }


def _parse_response(*, client: Client, response: httpx.Response) -> Optional[Any]:
    if response.status_code == HTTPStatus.OK:
        return None
    if response.status_code == HTTPStatus.BAD_REQUEST:
        return None
    if response.status_code == HTTPStatus.FORBIDDEN:
        return None
    if response.status_code == HTTPStatus.NOT_FOUND:
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
    id: str,
    *,
    client: Client,
    json_body: UpdatePrivateLinkJsonBody,
) -> Response[Any]:
    """Updates a given private link for an annotation for zarr streaming
    Expects:
     - As JSON object body with keys:
      - annotation (string): annotation id to create private link for
      - expirationDateTime (Optional[bool]): optional UNIX timestamp, expiration date and time for the
    link

    Args:
        id (str):
        json_body (UpdatePrivateLinkJsonBody):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any]
    """

    kwargs = _get_kwargs(
        id=id,
        client=client,
        json_body=json_body,
    )

    response = httpx.request(
        verify=client.verify_ssl,
        **kwargs,
    )

    return _build_response(client=client, response=response)


async def asyncio_detailed(
    id: str,
    *,
    client: Client,
    json_body: UpdatePrivateLinkJsonBody,
) -> Response[Any]:
    """Updates a given private link for an annotation for zarr streaming
    Expects:
     - As JSON object body with keys:
      - annotation (string): annotation id to create private link for
      - expirationDateTime (Optional[bool]): optional UNIX timestamp, expiration date and time for the
    link

    Args:
        id (str):
        json_body (UpdatePrivateLinkJsonBody):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any]
    """

    kwargs = _get_kwargs(
        id=id,
        client=client,
        json_body=json_body,
    )

    async with httpx.AsyncClient(verify=client.verify_ssl) as _client:
        response = await _client.request(**kwargs)

    return _build_response(client=client, response=response)
