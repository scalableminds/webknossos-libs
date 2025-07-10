import json

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
    ApiProjectCreate,
    ApiSharingToken,
    ApiShortLink,
    ApiTask,
    ApiTaskCreationResult,
    ApiTaskParameters,
    ApiTaskType,
    ApiTaskTypeCreate,
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
        headers: dict[str, str] | None = None,
    ):
        super().__init__(timeout_seconds, headers)
        self.base_wk_url = base_wk_url.rstrip("/")

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
        sharing_token: str | None = None,
    ) -> ApiDataset:
        route = f"/datasets/{dataset_id}"
        return self._get_json(route, ApiDataset, query={"sharingToken": sharing_token})

    def dataset_id_from_name(self, directory_name: str, organization_id: str) -> str:
        route = f"/datasets/disambiguate/{organization_id}/{directory_name}/toId"
        return self._get_json(route, ApiDatasetId).id

    def dataset_list(
        self,
        is_active: bool | None,
        organization_id: str | None,
        name: str | None,
        folder_id: str | None,
    ) -> list[ApiDataset]:
        route = "/datasets"
        return self._get_json(
            route,
            list[ApiDataset],
            query={
                "isActive": is_active,
                "organizationId": organization_id,
                "searchQuery": name,
                "folderId": folder_id,
            },
        )

    def dataset_update_teams(self, dataset_id: str, team_ids: list[str]) -> None:
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

    def annotation_list(self, is_finished: bool | None) -> list[ApiAnnotation]:
        route = "/annotations/readable"
        return self._get_json(
            route,
            list[ApiAnnotation],
            query={
                "isFinished": is_finished,
            },
        )

    def datastore_list(self) -> list[ApiDataStore]:
        route = "/datastores"
        return self._get_json(route, list[ApiDataStore])

    def project_create(self, project: ApiProjectCreate) -> ApiProject:
        route = "/projects"
        return self._post_json_with_json_response(route, project, ApiProject)

    def project_delete(self, project_id: str) -> None:
        route = f"/projects/{project_id}"
        self._delete(route)

    def project_update(self, project_id: str, project: ApiProjectCreate) -> ApiProject:
        route = f"/projects/{project_id}"
        return self._put_json_with_json_response(route, project, ApiProject)

    def project_info_by_name(self, project_name: str) -> ApiProject:
        route = f"/projects/byName/{project_name}"
        return self._get_json(route, ApiProject)

    def project_info_by_id(self, project_id: str) -> ApiProject:
        route = f"/projects/{project_id}"
        return self._get_json(route, ApiProject)

    def task_infos_by_project_id_paginated(
        self, project_id: str, limit: int | None, page_number: int | None
    ) -> tuple[list[ApiTask], int]:
        route = f"/projects/{project_id}/tasks"
        return self._get_json_paginated(route, list[ApiTask], limit, page_number)

    def task_type_create(self, task_type: ApiTaskTypeCreate) -> ApiTaskType:
        route = "/taskTypes"
        return self._post_json_with_json_response(route, task_type, ApiTaskType)

    def task_type_delete(self, task_type_id: str) -> None:
        route = f"/taskTypes/{task_type_id}"
        self._delete(route)

    def task_type_list(self) -> list[ApiTaskType]:
        route = "/taskTypes"
        return self._get_json(route, list[ApiTaskType])

    def get_task_type(self, task_type_id: str) -> ApiTaskType:
        route = f"/taskTypes/{task_type_id}"
        return self._get_json(route, ApiTaskType)

    def annotation_info(self, annotation_id: str) -> ApiAnnotation:
        route = f"/annotations/{annotation_id}/info"
        return self._get_json(
            route, ApiAnnotation, query={"timestamp": time_since_epoch_in_ms()}
        )

    def annotation_download(
        self, annotation_id: str, skip_volume_data: bool, retry_count: int = 0
    ) -> tuple[bytes, str]:
        route = f"/annotations/{annotation_id}/download"
        return self._get_file(
            route, query={"skipVolumeData": skip_volume_data}, retry_count=retry_count
        )

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

    def annotation_edit(
        self, annotation_typ: str, annotation_id: str, annotation: ApiAnnotation
    ) -> None:
        route = f"/annotations/{annotation_typ}/{annotation_id}/edit"
        self._patch_json(route, annotation)

    def annotation_infos_by_task(self, task_id: str) -> list[ApiAnnotation]:
        route = f"/tasks/{task_id}/annotations"
        return self._get_json(route, list[ApiAnnotation])

    def task_info(self, task_id: str) -> ApiTask:
        route = f"/tasks/{task_id}"
        return self._get_json(route, ApiTask)

    def task_list(self) -> list[ApiTask]:
        route = "/tasks/list"
        return self._post_json_with_json_response(route, {}, list[ApiTask])

    def folder_tree(self) -> list[ApiFolderWithParent]:
        route = "/folders/tree"
        return self._get_json(route, list[ApiFolderWithParent])

    def user_by_id(self, user_id: str) -> ApiUser:
        route = f"/users/{user_id}"
        return self._get_json(route, ApiUser)

    def user_current(self) -> ApiUser:
        route = "/user"
        return self._get_json(route, ApiUser)

    def user_list(self) -> list[ApiUser]:
        route = "/users"
        return self._get_json(route, list[ApiUser])

    def user_logged_time(self, user_id: str) -> ApiLoggedTimeGroupedByMonth:
        route = f"/users/{user_id}/loggedTime"
        return self._get_json(route, ApiLoggedTimeGroupedByMonth)

    def user_update(self, user: ApiUser) -> None:
        route = f"/users/{user.id}"
        self._patch_json(route, user)

    def team_list(self) -> list[ApiTeam]:
        route = "/teams"
        return self._get_json(route, list[ApiTeam])

    def team_add(self, team: ApiTeamAdd) -> None:
        route = "/teams"
        self._post_json(route, team)

    def token_generate_for_data_store(self) -> ApiDataStoreToken:
        route = "/userToken/generate"
        return self._post_with_json_response(route, ApiDataStoreToken)

    def tasks_create(
        self, task_parameters: list[ApiTaskParameters]
    ) -> ApiTaskCreationResult:
        route = "/tasks"
        response = self._post_json_with_json_response(
            route, task_parameters, ApiTaskCreationResult
        )
        return response

    def task_update(self, task_id: str, task_parameters: ApiTaskParameters) -> ApiTask:
        route = f"/tasks/{task_id}"
        return self._put_json_with_json_response(route, task_parameters, ApiTask)

    def task_delete(self, task_id: str) -> None:
        route = f"/tasks/{task_id}"
        self._delete(route)

    def tasks_create_from_files(
        self,
        nml_task_parameters: ApiNmlTaskParameters,
        annotation_files: list[tuple[str, bytes]],
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
