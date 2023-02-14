from http import HTTPStatus
from typing import Any, Dict

import httpx

from ...client import Client
from ...models.task_create_from_files_json_body import TaskCreateFromFilesJsonBody
from ...types import Response


def _get_kwargs(
    *,
    client: Client,
    json_body: TaskCreateFromFilesJsonBody,
) -> Dict[str, Any]:
    url = "{}/api/tasks/createFromFiles".format(client.base_url)

    headers: Dict[str, str] = client.get_headers()
    cookies: Dict[str, Any] = client.get_cookies()

    json_json_body = json_body.to_dict()

    return {
        "method": "post",
        "url": url,
        "headers": headers,
        "cookies": cookies,
        "timeout": client.get_timeout(),
        "json": json_json_body,
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
    json_body: TaskCreateFromFilesJsonBody,
) -> Response[Any]:
    """Create new tasks from existing annotation files
    Expects:
     - As Form data:
       - taskTypeId (string) id of the task type to be used for the new tasks
       - neededExperience (Experience) experience domain and level that selects which users can get the
    new tasks
       - openInstances (int) if greater than one, multiple instances of the task will be given to users
    to annotate
       - projectName (string) name of the project the task should be part of
       - scriptId (string, optional) id of a user script that should be loaded for the annotators of the
    new tasks
       - boundingBox (BoundingBox, optional) limit the bounding box where the annotators should be
    active
     - As File attachment
       - A zip file containing base annotations (each either NML or zip with NML + volume) for the new
    tasks. One task will be created per annotation.

    Args:
        json_body (TaskCreateFromFilesJsonBody):

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

    return _build_response(response=response)


async def asyncio_detailed(
    *,
    client: Client,
    json_body: TaskCreateFromFilesJsonBody,
) -> Response[Any]:
    """Create new tasks from existing annotation files
    Expects:
     - As Form data:
       - taskTypeId (string) id of the task type to be used for the new tasks
       - neededExperience (Experience) experience domain and level that selects which users can get the
    new tasks
       - openInstances (int) if greater than one, multiple instances of the task will be given to users
    to annotate
       - projectName (string) name of the project the task should be part of
       - scriptId (string, optional) id of a user script that should be loaded for the annotators of the
    new tasks
       - boundingBox (BoundingBox, optional) limit the bounding box where the annotators should be
    active
     - As File attachment
       - A zip file containing base annotations (each either NML or zip with NML + volume) for the new
    tasks. One task will be created per annotation.

    Args:
        json_body (TaskCreateFromFilesJsonBody):

    Returns:
        Response[Any]
    """

    kwargs = _get_kwargs(
        client=client,
        json_body=json_body,
    )

    async with httpx.AsyncClient(verify=client.verify_ssl) as _client:
        response = await _client.request(**kwargs)

    return _build_response(response=response)
