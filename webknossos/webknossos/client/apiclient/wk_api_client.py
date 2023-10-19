from .abstract_api_client import AbstractApiClient
from typing import Any, Dict, List, Optional

from webknossos.client.apiclient.models import (
    ApiDataset,
    ApiSharingToken,
    ApiShortLink
)


class WkApiClient(AbstractApiClient):

    def short_link_by_key(self, key: str) -> ApiShortLink:
        uri = f"{self._api_uri}/shortLinks/byKey/{key}"
        return self._get_json(uri, ApiShortLink)

    def dataset_info(
            self, organization_name, dataset_name, sharing_token: Optional[str]
    ) -> ApiDataset:
        uri = f"{self._api_uri}/datasets/{organization_name}/{dataset_name}"
        return self._get_json(uri, ApiDataset, query={"sharing_token": sharing_token})

    def dataset_list(
            self, is_active: Optional[bool], organization_name: Optional[str]
    ) -> List[ApiDataset]:
        uri = f"{self._api_uri}/datasets"
        return self._get_json(
            uri,
            List[ApiDataset],
            query={"isActive": is_active, "organizationName": organization_name},
        )

    def dataset_update_teams(
            self, organization_name: str, dataset_name: str, team_ids: List[str]
    ) -> None:
        uri = f"{self._api_uri}/datasets/{organization_name}/{dataset_name}/teams"
        self._patch_json(uri, team_ids)

    def dataset_update(
            self, organization_name: str, dataset_name: str, updated_dataset: ApiDataset
    ) -> None:
        uri = f"{self._api_uri}/datasets/{organization_name}/{dataset_name}"
        self._patch_json(uri, updated_dataset)

    def dataset_sharing_token(
            self, organization_name: str, dataset_name: str
    ) -> ApiSharingToken:
        uri = (
            f"{self._api_uri}/datasets/{organization_name}/{dataset_name}/sharingToken"
        )
        return self._get_json(uri, ApiSharingToken)