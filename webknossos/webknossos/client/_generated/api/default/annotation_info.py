from typing import Any, Dict, Optional

import httpx

from ...client import Client
from ...models.annotation_info_response_200 import AnnotationInfoResponse200
from ...types import UNSET, Response


def _get_kwargs(
    typ: str,
    id: str,
    *,
    client: Client,
    timestamp: int,
) -> Dict[str, Any]:
    url = "{}/api/annotations/{typ}/{id}/info".format(client.base_url, typ=typ, id=id)

    headers: Dict[str, Any] = client.get_headers()
    cookies: Dict[str, Any] = client.get_cookies()

    params: Dict[str, Any] = {
        "timestamp": timestamp,
    }
    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    return {
        "url": url,
        "headers": headers,
        "cookies": cookies,
        "timeout": client.get_timeout(),
        "params": params,
    }


def _parse_response(*, response: httpx.Response) -> Optional[AnnotationInfoResponse200]:
    if response.status_code == 200:
        response_200 = AnnotationInfoResponse200.from_dict(response.json())

        return response_200
    return None


def _build_response(*, response: httpx.Response) -> Response[AnnotationInfoResponse200]:
    return Response(
        status_code=response.status_code,
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(response=response),
    )


def sync_detailed(
    typ: str,
    id: str,
    *,
    client: Client,
    timestamp: int,
) -> Response[AnnotationInfoResponse200]:
    kwargs = _get_kwargs(
        typ=typ,
        id=id,
        client=client,
        timestamp=timestamp,
    )

    response = httpx.get(
        **kwargs,
    )

    return _build_response(response=response)


def sync(
    typ: str,
    id: str,
    *,
    client: Client,
    timestamp: int,
) -> Optional[AnnotationInfoResponse200]:
    """ """

    return sync_detailed(
        typ=typ,
        id=id,
        client=client,
        timestamp=timestamp,
    ).parsed


async def asyncio_detailed(
    typ: str,
    id: str,
    *,
    client: Client,
    timestamp: int,
) -> Response[AnnotationInfoResponse200]:
    kwargs = _get_kwargs(
        typ=typ,
        id=id,
        client=client,
        timestamp=timestamp,
    )

    async with httpx.AsyncClient() as _client:
        response = await _client.get(**kwargs)

    return _build_response(response=response)


async def asyncio(
    typ: str,
    id: str,
    *,
    client: Client,
    timestamp: int,
) -> Optional[AnnotationInfoResponse200]:
    """ """

    return (
        await asyncio_detailed(
            typ=typ,
            id=id,
            client=client,
            timestamp=timestamp,
        )
    ).parsed
