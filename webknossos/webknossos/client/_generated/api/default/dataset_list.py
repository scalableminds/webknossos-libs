from typing import Any, Dict, List, Optional, Union

import httpx

from ...client import Client
from ...models.dataset_list_response_200_item import DatasetListResponse200Item
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    client: Client,
    is_active: Union[Unset, None, bool] = UNSET,
    is_unreported: Union[Unset, None, bool] = UNSET,
    is_editable: Union[Unset, None, bool] = UNSET,
    organization_name: Union[Unset, None, str] = UNSET,
    only_my_organization: Union[Unset, None, bool] = UNSET,
    uploader_id: Union[Unset, None, str] = UNSET,
    folder_id: Union[Unset, None, str] = UNSET,
    search_query: Union[Unset, None, str] = UNSET,
    limit: Union[Unset, None, int] = UNSET,
) -> Dict[str, Any]:
    url = "{}/api/datasets".format(client.base_url)

    headers: Dict[str, Any] = client.get_headers()
    cookies: Dict[str, Any] = client.get_cookies()

    params: Dict[str, Any] = {
        "isActive": is_active,
        "isUnreported": is_unreported,
        "isEditable": is_editable,
        "organizationName": organization_name,
        "onlyMyOrganization": only_my_organization,
        "uploaderId": uploader_id,
        "folderId": folder_id,
        "searchQuery": search_query,
        "limit": limit,
    }
    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    return {
        "url": url,
        "headers": headers,
        "cookies": cookies,
        "timeout": client.get_timeout(),
        "params": params,
    }


def _parse_response(
    *, response: httpx.Response
) -> Optional[List[DatasetListResponse200Item]]:
    if response.status_code == 200:
        response_200 = []
        _response_200 = response.json()
        for response_200_item_data in _response_200:
            response_200_item = DatasetListResponse200Item.from_dict(
                response_200_item_data
            )

            response_200.append(response_200_item)

        return response_200
    return None


def _build_response(
    *, response: httpx.Response
) -> Response[List[DatasetListResponse200Item]]:
    return Response(
        status_code=response.status_code,
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(response=response),
    )


def sync_detailed(
    *,
    client: Client,
    is_active: Union[Unset, None, bool] = UNSET,
    is_unreported: Union[Unset, None, bool] = UNSET,
    is_editable: Union[Unset, None, bool] = UNSET,
    organization_name: Union[Unset, None, str] = UNSET,
    only_my_organization: Union[Unset, None, bool] = UNSET,
    uploader_id: Union[Unset, None, str] = UNSET,
    folder_id: Union[Unset, None, str] = UNSET,
    search_query: Union[Unset, None, str] = UNSET,
    limit: Union[Unset, None, int] = UNSET,
) -> Response[List[DatasetListResponse200Item]]:
    kwargs = _get_kwargs(
        client=client,
        is_active=is_active,
        is_unreported=is_unreported,
        is_editable=is_editable,
        organization_name=organization_name,
        only_my_organization=only_my_organization,
        uploader_id=uploader_id,
        folder_id=folder_id,
        search_query=search_query,
        limit=limit,
    )

    response = httpx.get(
        **kwargs,
    )

    return _build_response(response=response)


def sync(
    *,
    client: Client,
    is_active: Union[Unset, None, bool] = UNSET,
    is_unreported: Union[Unset, None, bool] = UNSET,
    is_editable: Union[Unset, None, bool] = UNSET,
    organization_name: Union[Unset, None, str] = UNSET,
    only_my_organization: Union[Unset, None, bool] = UNSET,
    uploader_id: Union[Unset, None, str] = UNSET,
    folder_id: Union[Unset, None, str] = UNSET,
    search_query: Union[Unset, None, str] = UNSET,
    limit: Union[Unset, None, int] = UNSET,
) -> Optional[List[DatasetListResponse200Item]]:
    """ """

    return sync_detailed(
        client=client,
        is_active=is_active,
        is_unreported=is_unreported,
        is_editable=is_editable,
        organization_name=organization_name,
        only_my_organization=only_my_organization,
        uploader_id=uploader_id,
        folder_id=folder_id,
        search_query=search_query,
        limit=limit,
    ).parsed


async def asyncio_detailed(
    *,
    client: Client,
    is_active: Union[Unset, None, bool] = UNSET,
    is_unreported: Union[Unset, None, bool] = UNSET,
    is_editable: Union[Unset, None, bool] = UNSET,
    organization_name: Union[Unset, None, str] = UNSET,
    only_my_organization: Union[Unset, None, bool] = UNSET,
    uploader_id: Union[Unset, None, str] = UNSET,
    folder_id: Union[Unset, None, str] = UNSET,
    search_query: Union[Unset, None, str] = UNSET,
    limit: Union[Unset, None, int] = UNSET,
) -> Response[List[DatasetListResponse200Item]]:
    kwargs = _get_kwargs(
        client=client,
        is_active=is_active,
        is_unreported=is_unreported,
        is_editable=is_editable,
        organization_name=organization_name,
        only_my_organization=only_my_organization,
        uploader_id=uploader_id,
        folder_id=folder_id,
        search_query=search_query,
        limit=limit,
    )

    async with httpx.AsyncClient() as _client:
        response = await _client.get(**kwargs)

    return _build_response(response=response)


async def asyncio(
    *,
    client: Client,
    is_active: Union[Unset, None, bool] = UNSET,
    is_unreported: Union[Unset, None, bool] = UNSET,
    is_editable: Union[Unset, None, bool] = UNSET,
    organization_name: Union[Unset, None, str] = UNSET,
    only_my_organization: Union[Unset, None, bool] = UNSET,
    uploader_id: Union[Unset, None, str] = UNSET,
    folder_id: Union[Unset, None, str] = UNSET,
    search_query: Union[Unset, None, str] = UNSET,
    limit: Union[Unset, None, int] = UNSET,
) -> Optional[List[DatasetListResponse200Item]]:
    """ """

    return (
        await asyncio_detailed(
            client=client,
            is_active=is_active,
            is_unreported=is_unreported,
            is_editable=is_editable,
            organization_name=organization_name,
            only_my_organization=only_my_organization,
            uploader_id=uploader_id,
            folder_id=folder_id,
            search_query=search_query,
            limit=limit,
        )
    ).parsed
