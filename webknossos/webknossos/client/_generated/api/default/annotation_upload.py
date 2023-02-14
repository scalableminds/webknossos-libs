from http import HTTPStatus
from typing import Any, Dict

import httpx

from ...client import Client
from ...types import Response


def _get_kwargs(
    *,
    client: Client,
) -> Dict[str, Any]:
    url = "{}/api/annotations/upload".format(client.base_url)

    headers: Dict[str, str] = client.get_headers()
    cookies: Dict[str, Any] = client.get_cookies()

    return {
        "method": "post",
        "url": url,
        "headers": headers,
        "cookies": cookies,
        "timeout": client.get_timeout(),
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
) -> Response[Any]:
    """Upload NML(s) or ZIP(s) of NML(s) to create a new explorative annotation.
    Expects:
     - As file attachment:
        - Any number of NML files or ZIP files containing NMLs, optionally with volume data ZIPs
    referenced from an NML in a ZIP
        - If multiple annotations are uploaded, they are merged into one.
           - This is not supported if any of the annotations has multiple volume layers.
     - As form parameter: createGroupForEachFile [String] should be one of \"true\" or \"false\"
       - If \"true\": in merged annotation, create tree group wrapping the trees of each file
       - If \"false\": in merged annotation, rename trees with the respective file name as prefix

    Returns:
        Response[Any]
    """

    kwargs = _get_kwargs(
        client=client,
    )

    response = httpx.request(
        verify=client.verify_ssl,
        **kwargs,
    )

    return _build_response(response=response)


async def asyncio_detailed(
    *,
    client: Client,
) -> Response[Any]:
    """Upload NML(s) or ZIP(s) of NML(s) to create a new explorative annotation.
    Expects:
     - As file attachment:
        - Any number of NML files or ZIP files containing NMLs, optionally with volume data ZIPs
    referenced from an NML in a ZIP
        - If multiple annotations are uploaded, they are merged into one.
           - This is not supported if any of the annotations has multiple volume layers.
     - As form parameter: createGroupForEachFile [String] should be one of \"true\" or \"false\"
       - If \"true\": in merged annotation, create tree group wrapping the trees of each file
       - If \"false\": in merged annotation, rename trees with the respective file name as prefix

    Returns:
        Response[Any]
    """

    kwargs = _get_kwargs(
        client=client,
    )

    async with httpx.AsyncClient(verify=client.verify_ssl) as _client:
        response = await _client.request(**kwargs)

    return _build_response(response=response)
