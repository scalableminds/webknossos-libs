from typing import Any, Dict

import httpx

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
    id: str,
    *,
    client: Client,
    json_body: UpdatePrivateLinkJsonBody,
) -> Response[Any]:
    kwargs = _get_kwargs(
        id=id,
        client=client,
        json_body=json_body,
    )

    response = httpx.put(
        **kwargs,
    )

    return _build_response(response=response)


async def asyncio_detailed(
    id: str,
    *,
    client: Client,
    json_body: UpdatePrivateLinkJsonBody,
) -> Response[Any]:
    kwargs = _get_kwargs(
        id=id,
        client=client,
        json_body=json_body,
    )

    async with httpx.AsyncClient() as _client:
        response = await _client.put(**kwargs)

    return _build_response(response=response)
