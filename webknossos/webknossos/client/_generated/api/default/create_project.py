from http import HTTPStatus
from typing import Any, Dict, Optional

import httpx

from ... import errors
from ...client import Client
from ...models.create_project_json_body import CreateProjectJsonBody
from ...types import Response


def _get_kwargs(
    *,
    client: Client,
    json_body: CreateProjectJsonBody,
) -> Dict[str, Any]:
    url = "{}/api/projects".format(client.base_url)

    headers: Dict[str, str] = client.get_headers()
    cookies: Dict[str, Any] = client.get_cookies()

    json_json_body = json_body.to_dict()

    return {
        "method": "post",
        "url": url,
        "headers": headers,
        "cookies": cookies,
        "timeout": client.get_timeout(),
        "follow_redirects": client.follow_redirects,
        "json": json_json_body,
    }


def _parse_response(*, client: Client, response: httpx.Response) -> Optional[Any]:
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
    json_body: CreateProjectJsonBody,
) -> Response[Any]:
    """Create a new Project.
    Expects:
     - As JSON object body with keys:
      - name (string) name of the new project
      - team (string) id of the team this project is for
      - priority (int) priority of the project’s tasks
      - paused (bool, default=False) if the project should be paused at time of creation (its tasks are
    not distributed)
      - expectedTime (int, optional) time limit
      - owner (string) id of a user
      - isBlacklistedFromReport (boolean) if true, the project is skipped on the progress report tables

    Args:
        json_body (CreateProjectJsonBody):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any]
    """

    kwargs = _get_kwargs(
        client=client,
        json_body=json_body,
    )

    response = httpx.request(
        verify=client.verify_ssl,
        **kwargs,
    )

    return _build_response(client=client, response=response)


async def asyncio_detailed(
    *,
    client: Client,
    json_body: CreateProjectJsonBody,
) -> Response[Any]:
    """Create a new Project.
    Expects:
     - As JSON object body with keys:
      - name (string) name of the new project
      - team (string) id of the team this project is for
      - priority (int) priority of the project’s tasks
      - paused (bool, default=False) if the project should be paused at time of creation (its tasks are
    not distributed)
      - expectedTime (int, optional) time limit
      - owner (string) id of a user
      - isBlacklistedFromReport (boolean) if true, the project is skipped on the progress report tables

    Args:
        json_body (CreateProjectJsonBody):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any]
    """

    kwargs = _get_kwargs(
        client=client,
        json_body=json_body,
    )

    async with httpx.AsyncClient(verify=client.verify_ssl) as _client:
        response = await _client.request(**kwargs)

    return _build_response(client=client, response=response)
