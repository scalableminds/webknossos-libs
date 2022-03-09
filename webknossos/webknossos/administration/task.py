import json
import logging
from typing import TYPE_CHECKING, BinaryIO, Dict, List, Mapping, Optional, Tuple, Union

import attr
import httpx

from webknossos.administration import Project
from webknossos.annotation import Annotation, AnnotationInfo
from webknossos.client._generated.api.default import (
    annotation_infos_by_task_id,
    task_info,
)
from webknossos.client.context import _get_generated_client
from webknossos.geometry import BoundingBox, Vec3Int

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from webknossos.client._generated.models.task_info_response_200 import (
        TaskInfoResponse200,
    )
    from webknossos.client._generated.models.task_infos_by_project_id_response_200_item import (
        TaskInfosByProjectIdResponse200Item,
    )


@attr.frozen
class TaskStatus:
    open_instance_count: int
    active_instance_count: int
    finished_instance_count: int


@attr.frozen
class Task:
    """Data class containing information about a webKnossos task"""

    task_id: str
    project_id: str
    dataset_name: str
    status: TaskStatus

    @classmethod
    def get_by_id(cls, task_id: str) -> "Task":
        """Returns the task specified by the passed id (if your token authorizes you to see it)"""
        client = _get_generated_client(enforce_auth=True)
        response = task_info.sync(id=task_id, client=client)
        assert (
            response is not None
        ), f"Requesting task infos from {client.base_url} failed."
        return cls._from_generated_response(response)

    @classmethod
    def create_from_annotations(
        cls,
        task_type_id: str,
        project_name: str,
        base_annotations: List[Annotation],
        needed_experience_domain: str,
        needed_experience_value: int,
        instances: int = 1,
        script_id: Optional[str] = None,
        bounding_box: Optional[BoundingBox] = None,
    ) -> List["Task"]:
        """Submits tasks in webKnossos based on existing annotations, and returns the Task objects"""

        assert (
            len(base_annotations) > 0
        ), "Must supply at least one base annotation to create tasks"

        client = _get_generated_client(enforce_auth=True)
        url = f"{client.base_url}/api/tasks/createFromFiles"
        task_parameters = {
            "taskTypeId": task_type_id,
            "neededExperience": {
                "domain": needed_experience_domain,
                "value": needed_experience_value,
            },
            "openInstances": instances,
            "projectName": project_name,
            "scriptId": script_id,
            "boundingBox": bounding_box.to_wkw_dict()
            if bounding_box is not None
            else None,
        }
        form_data = {"formJSON": json.dumps(task_parameters)}
        files: Mapping[str, Tuple[str, Union[bytes, BinaryIO]]] = {
            f"{a.name}.zip": (f"{a.name}.zip", a._binary_zip())
            for a in base_annotations
        }

        response = httpx.post(
            url=url,
            headers=client.get_headers(),
            cookies=client.get_cookies(),
            timeout=client.get_timeout(),
            data=form_data,
            files=files,
        )
        assert (
            response.status_code == 200
        ), f"Failed to create tasks from files: {response.status_code}: {response.text}"

        return cls._handle_task_creation_response(response)

    @classmethod
    def create(
        cls,
        task_type_id: str,
        project_name: str,
        dataset_name: str,
        needed_experience_domain: str,
        needed_experience_value: int,
        starting_position: Vec3Int,
        starting_rotation: Optional[Vec3Int] = Vec3Int(0, 0, 0),
        instances: int = 1,
        script_id: Optional[str] = None,
        bounding_box: Optional[BoundingBox] = None,
    ) -> List["Task"]:
        """Submits tasks in webKnossos based on a dataset, starting position + rotation, and returns the Task objects"""

        client = _get_generated_client(enforce_auth=True)
        url = f"{client.base_url}/api/tasks"
        task_parameters = {
            "taskTypeId": task_type_id,
            "neededExperience": {
                "domain": needed_experience_domain,
                "value": needed_experience_value,
            },
            "openInstances": instances,
            "projectName": project_name,
            "scriptId": script_id,
            "dataSet": dataset_name,
            "editPosition": starting_position,
            "editRotation": starting_rotation,
            "boundingBox": bounding_box.to_wkw_dict()
            if bounding_box is not None
            else None,
        }

        response = httpx.post(
            url=url,
            headers=client.get_headers(),
            cookies=client.get_cookies(),
            timeout=client.get_timeout(),
            json=[task_parameters],
        )
        assert (
            response.status_code == 200
        ), f"Failed to create tasks: {response.status_code}: {response.text}"

        return cls._handle_task_creation_response(response)

    @classmethod
    def _from_dict(cls, response_dict: Dict) -> "Task":
        from webknossos.client._generated.models.task_info_response_200 import (
            TaskInfoResponse200,
        )

        return cls._from_generated_response(
            TaskInfoResponse200.from_dict(response_dict)
        )

    @classmethod
    def _from_generated_response(
        cls,
        response: Union["TaskInfoResponse200", "TaskInfosByProjectIdResponse200Item"],
    ) -> "Task":
        return cls(
            response.id,
            response.project_name,
            response.data_set,
            TaskStatus(
                response.status.open_, response.status.active, response.status.finished
            ),
        )

    def get_annotation_infos(self) -> List[AnnotationInfo]:
        """Returns AnnotationInfo objects describing all task instances that have been started by annotators for this task"""
        client = _get_generated_client(enforce_auth=True)
        response = annotation_infos_by_task_id.sync(id=self.task_id, client=client)
        assert (
            response is not None
        ), f"Requesting annotation infos for task from {client.base_url} failed."
        return [AnnotationInfo._from_generated_response(a) for a in response]

    def get_project(self) -> Project:
        """Returns the project this task belongs to"""
        return Project.get_by_id(self.project_id)

    @classmethod
    def _handle_task_creation_response(cls, response: httpx.Response) -> List["Task"]:
        result = response.json()
        if "warnings" in result:
            warnings = result["warnings"]
            if len(warnings) > 0:
                logger.warning(
                    f"There were {len(warnings)} warnings during task creation:"
                )
            for warning in warnings:
                logger.warning(warning)
        assert "tasks" in result, "Invalid result of task creation"

        successes = []
        errors = []
        for t in result["tasks"]:
            print(t)
            if "success" in t:
                successes.append(t["success"])
            if "error" in t:
                errors.append(t["error"])
        if len(errors) > 0:
            logger.error(f"{len(errors)} tasks could not be created:")
            for error in errors:
                logger.error(error)
        if len(successes) > 0:
            logger.info(f"{len(successes)} tasks were successfully created.")
        return [cls._from_dict(t) for t in successes]
