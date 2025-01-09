import json
from typing import Dict, List, Optional, Tuple

import httpx

from webknossos.client.api_client.models import (
    ApiAnnotation,
    ApiAnnotationUploadResult,
    ApiDataset,
    ApiDatasetExploreAndAddRemote,
    ApiDatasetId,
    ApiDatasetIsValidNewNameResponse,
    ApiDataStore,
    ApiDataStoreToken,
    ApiFolderWithParent,
    ApiLoggedTimeGroupedByMonth,
    ApiNmlTaskParameters,
    ApiProject,
    ApiSharingToken,
    ApiShortLink,
    ApiTask,
    ApiTaskCreationResult,
    ApiTaskParameters,
    ApiTeam,
    ApiTeamAdd,
    ApiUser,
    ApiWkBuildInfo,
)

from ...utils import time_since_epoch_in_ms
from ._abstract_api_client import AbstractApiClient


class WkApiClient(AbstractApiClient):
    # Client to use the HTTP API of WEBKNOSSOS servers.
    # When adding a method here, use the utility methods from AbstractApiClient
    # and add more as needed.
    # Methods here are prefixed with the domain, e.g. dataset_update_teams (not update_dataset_teams)

    def __init__(
        self,
        base_wk_url: str,
        timeout_seconds: float,
        headers: Optional[Dict[str, str]] = None,
        webknossos_api_version: int = 9,
    ):
        super().__init__(timeout_seconds, headers)
        self.webknossos_api_version = webknossos_api_version
        self.base_wk_url = base_wk_url

    @property
    def url_prefix(self) -> str:
        return f"{self.base_wk_url}/api/v{self.webknossos_api_version}"

    def health(self) -> None:
        route = "/health"
        self._get(route)

    def build_info(self) -> ApiWkBuildInfo:
        route = "/buildinfo"
        return self._get_json(route, ApiWkBuildInfo)

    def short_link_by_key(self, key: str) -> ApiShortLink:
        route = f"/shortLinks/byKey/{key}"
        return self._get_json(route, ApiShortLink)

    def dataset_info(
        self,
        dataset_id: str,
        sharing_token: Optional[str] = None,
    ) -> ApiDataset:
        route = f"/datasets/{dataset_id}"
        return self._get_json(route, ApiDataset, query={"sharingToken": sharing_token})

    def dataset_id_from_name(self, dataset_name: str, organization_id: str) -> str:
        route = f"/datasets/disambiguate/{organization_id}/{dataset_name}/toId"
        return self._get_json(route, ApiDatasetId).id

    def dataset_list(
        self,
        is_active: Optional[bool],
        organization_id: Optional[str],
        name: Optional[str],
        folder_id: Optional[str],
    ) -> List[ApiDataset]:
        route = "/datasets"
        return self._get_json(
            route,
            List[ApiDataset],
            query={
                "isActive": is_active,
                "organizationId": organization_id,
                "searchQuery": name,
                "folderId": folder_id,
            },
        )

    def dataset_update_teams(self, dataset_id: str, team_ids: List[str]) -> None:
        route = f"/datasets/{dataset_id}/teams"
        self._patch_json(route, team_ids)

    def dataset_update(self, dataset_id: str, updated_dataset: ApiDataset) -> None:
        route = f"/datasets/{dataset_id}"
        self._patch_json(route, updated_dataset)

    def dataset_sharing_token(self, dataset_id: str) -> ApiSharingToken:
        route = f"/datasets/{dataset_id}/sharingToken"
        return self._get_json(route, ApiSharingToken)

    def dataset_is_valid_new_name(
        self, dataset_name: str
    ) -> ApiDatasetIsValidNewNameResponse:
        route = f"/datasets/{dataset_name}/isValidNewName"
        return self._get_json(route, ApiDatasetIsValidNewNameResponse)

    def dataset_explore_and_add_remote(
        self, dataset: ApiDatasetExploreAndAddRemote
    ) -> None:
        route = "/datasets/exploreAndAddRemote"
        self._post_json(
            route,
            dataset,
        )

    def datastore_list(self) -> List[ApiDataStore]:
        route = "/datastores"
        return self._get_json(route, List[ApiDataStore])

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

    def annotation_info(self, annotation_id: str) -> ApiAnnotation:
        route = f"/annotations/{annotation_id}/info"
        return self._get_json(
            route, ApiAnnotation, query={"timestamp": time_since_epoch_in_ms()}
        )

    def annotation_download(
        self, annotation_id: str, skip_volume_data: bool
    ) -> Tuple[bytes, str]:
        route = f"/annotations/{annotation_id}/download"
        return self._get_file(route, query={"skipVolumeData": skip_volume_data})

    def annotation_upload(
        self, file_body: bytes, filename: str, createGroupForEachFile: bool
    ) -> ApiAnnotationUploadResult:
        route = "/annotations/upload"
        data: httpx._types.RequestData = {
            "createGroupForEachFile": createGroupForEachFile
        }
        files: httpx._types.RequestFiles = {
            filename: (filename, file_body),
        }
        return self.post_multipart_with_json_response(
            route, ApiAnnotationUploadResult, data, files
        )

    def annotation_infos_by_task(self, task_id: str) -> List[ApiAnnotation]:
        route = f"/tasks/{task_id}/annotations"
        return self._get_json(route, List[ApiAnnotation])

    def task_info(self, task_id: str) -> ApiTask:
        route = f"/tasks/{task_id}"
        return self._get_json(route, ApiTask)

    def folder_tree(self) -> List[ApiFolderWithParent]:
        route = "/folders/tree"
        return self._get_json(route, List[ApiFolderWithParent])

    def user_by_id(self, user_id: str) -> ApiUser:
        route = f"/users/{user_id}"
        return self._get_json(route, ApiUser)

    def user_current(self) -> ApiUser:
        route = "/user"
        return self._get_json(route, ApiUser)

    def user_list(self) -> List[ApiUser]:
        route = "/users"
        return self._get_json(route, List[ApiUser])

    def user_logged_time(self, user_id: str) -> ApiLoggedTimeGroupedByMonth:
        route = f"/users/{user_id}/loggedTime"
        return self._get_json(route, ApiLoggedTimeGroupedByMonth)

    def user_update(self, user: ApiUser) -> None:
        route = f"/users/{user.id}"
        self._patch_json(route, user)

    def team_list(self) -> List[ApiTeam]:
        route = "/teams"
        return self._get_json(route, List[ApiTeam])

    def team_add(self, team: ApiTeamAdd) -> None:
        route = "/teams"
        self._post_json(route, team)

    def token_generate_for_data_store(self) -> ApiDataStoreToken:
        route = "/userToken/generate"
        return self._post_with_json_response(route, ApiDataStoreToken)

    def tasks_create(
        self, task_parameters: List[ApiTaskParameters]
    ) -> ApiTaskCreationResult:
        route = "/tasks"
        return self._post_json_with_json_response(
            route, task_parameters, ApiTaskCreationResult
        )

    def tasks_create_from_files(
        self,
        nml_task_parameters: ApiNmlTaskParameters,
        annotation_files: List[Tuple[str, bytes]],
    ) -> ApiTaskCreationResult:
        route = "/tasks/createFromFiles"
        data: httpx._types.RequestData = {
            "formJSON": json.dumps(self._prepare_for_json(nml_task_parameters))
        }
        files: httpx._types.RequestFiles = {
            filename: (filename, file_body) for filename, file_body in annotation_files
        }
        return self.post_multipart_with_json_response(
            route, ApiTaskCreationResult, multipart_data=data, files=files
        )
