from typing import Dict, List, Optional, Tuple

from webknossos.client.apiclient.models import (
    ApiAnnotation,
    ApiDataset,
    ApiDatastore,
    ApiProject,
    ApiSharingToken,
    ApiShortLink,
    ApiTask,
)

from .abstract_api_client import AbstractApiClient


class WkApiClient(AbstractApiClient):
    def __init__(
        self,
        base_wk_url: str,
        timeout_seconds: float,
        headers: Optional[Dict[str, str]] = None,
        webknossos_api_version: int = 5,
    ):
        super().__init__(timeout_seconds, headers)
        self.webknossos_api_version = webknossos_api_version
        self.base_wk_url = base_wk_url

    @property
    def url_prefix(self) -> str:
        return f"{self.base_wk_url}/api/v{self.webknossos_api_version}"

    def short_link_by_key(self, key: str) -> ApiShortLink:
        route = f"/shortLinks/byKey/{key}"
        return self._get_json(route, ApiShortLink)

    def dataset_info(
        self, organization_name: str, dataset_name: str, sharing_token: Optional[str]
    ) -> ApiDataset:
        route = f"/datasets/{organization_name}/{dataset_name}"
        return self._get_json(route, ApiDataset, query={"sharing_token": sharing_token})

    def dataset_list(
        self, is_active: Optional[bool], organization_name: Optional[str]
    ) -> List[ApiDataset]:
        route = "/datasets"
        return self._get_json(
            route,
            List[ApiDataset],
            query={"isActive": is_active, "organizationName": organization_name},
        )

    def dataset_update_teams(
        self, organization_name: str, dataset_name: str, team_ids: List[str]
    ) -> None:
        route = f"/datasets/{organization_name}/{dataset_name}/teams"
        self._patch_json(route, team_ids)

    def dataset_update(
        self, organization_name: str, dataset_name: str, updated_dataset: ApiDataset
    ) -> None:
        route = f"/datasets/{organization_name}/{dataset_name}"
        self._patch_json(route, updated_dataset)

    def dataset_sharing_token(
        self, organization_name: str, dataset_name: str
    ) -> ApiSharingToken:
        route = f"/datasets/{organization_name}/{dataset_name}/sharingToken"
        return self._get_json(route, ApiSharingToken)

    def assert_new_dataset_name_is_valid(
        self, organization_name: str, dataset_name: str
    ) -> None:
        route = f"/datasets/{organization_name}/{dataset_name}/isValidNewName"
        self._get(route)

    def datastore_list(self) -> List[ApiDatastore]:
        route = "/datastores"
        return self._get_json(route, List[ApiDatastore])

    def project_info_by_name(self, project_name: str) -> ApiProject:
        route = f"/projects/byName/{project_name}"
        return self._get_json(route, ApiProject)

    def project_info_by_id(self, project_id: str) -> ApiProject:
        route = f"/projects/{project_id}"
        return self._get_json(route, ApiProject)

    def task_infos_by_project_id_paginated(
        self, project_id: str, limit: Optional[int], page_number: Optional[int]
    ) -> Tuple[List[ApiTask], int]:
        route = f"/projects/{project_id}/tasks"
        return self._get_json_paginated(route, List[ApiTask], limit, page_number)

    def annotation_download(
        self, annotation_id: str, skip_volume_data: bool
    ) -> (bytes, str):
        route = f"/annotations/{annotation_id}/download"
        return self._get_file(route)

    def annotation_upload(
        self, file_body: bytes, filename: str, createGroupForEachFile: bool
    ) -> ApiAnnotation:
        route = "/annotations/upload"
        data = ({"createGroupForEachFile": createGroupForEachFile},)
        files = {
            filename: (filename, file_body),
        }
        return self.post_multipart_with_json_response(route, ApiAnnotation, data, files)

    def annotation_infos_by_task(self, task_id: str) -> List[ApiAnnotation]:
        route = f"/tasks/{task_id}/annotations"
        return self._get_json(self, List[ApiAnnotation])

    def task_info(self, task_id: str) -> ApiTask:
        route = f"/tasks/{task_id}"
        return self_get_json(self, ApiTask)