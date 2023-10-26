import json
import logging
from typing import TYPE_CHECKING, BinaryIO, Dict, List, Mapping, Optional, Tuple, Union

import attr
import httpx

from ..annotation import Annotation, AnnotationInfo
from ..client.apiclient.models import ApiTask, ApiTaskType, ApiTaskParameters, ApiNmlTaskParameters, ApiExperience, ApiBoundingBox, ApiTaskCreationResult
from ..client.context import _get_api_client
from ..dataset.dataset import RemoteDataset
from ..geometry import BoundingBox, Vec3Int
from ..utils import warn_deprecated
from .project import Project

logger = logging.getLogger(__name__)


@attr.frozen
class TaskStatus:
    pending_instance_count: int
    active_instance_count: int
    finished_instance_count: int

    @property
    def open_instance_count(self) -> int:
        warn_deprecated("open_instance_count", "pending_instance_count")
        return self.pending_instance_count


@attr.frozen
class TaskType:
    task_type_id: str
    name: str
    description: str
    team_id: str
    team_name: str

    @classmethod
    def _from_api_task_type(cls, api_task_type: ApiTaskType) -> "TaskType":
        return cls(
            api_task_type.id,
            api_task_type.summary,
            api_task_type.description,
            api_task_type.team_id,
            api_task_type.team_name,
        )


@attr.frozen
class Task:
    """Data class containing information about a WEBKNOSSOS task"""

    task_id: str
    project_id: str
    dataset_name: str
    status: TaskStatus
    task_type: TaskType

    @classmethod
    def get_by_id(cls, task_id: str) -> "Task":
        """Returns the task specified by the passed id (if your token authorizes you to see it)"""
        client = _get_api_client(enforce_auth=True)
        api_task = client.task_info(task_id)
        return cls._from_api_task(api_task)

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
        """Submits tasks in WEBKNOSSOS based on existing annotations, and returns the Task objects"""

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
            "pendingInstances": instances,
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

        return cls._handle_task_creation_result(response)

    @classmethod
    def create(
        cls,
        task_type_id: str,
        project_name: str,
        dataset_name: Union[str, RemoteDataset],
        needed_experience_domain: str,
        needed_experience_value: int,
        starting_position: Vec3Int,
        starting_rotation: Optional[Vec3Int] = Vec3Int(0, 0, 0),
        instances: int = 1,
        script_id: Optional[str] = None,
        bounding_box: Optional[BoundingBox] = None,
    ) -> List["Task"]:
        """Submits tasks in WEBKNOSSOS based on a dataset, starting position + rotation, and returns the Task objects"""

        client = _get_api_client(enforce_auth=True)
        if isinstance(dataset_name, RemoteDataset):
            dataset_name = dataset_name._dataset_name
        task_parameters = ApiTaskParameters(
            task_type_id = task_type_id,
            needed_experience= ApiExperience(
                domain= needed_experience_domain,
                value= needed_experience_value,
            ),
            pending_instances= instances,
            project_name=project_name,
            script_id=script_id,
            dataset_name= dataset_name,
            edit_position= starting_position,
            edit_rotation= starting_rotation,
            bounding_box= ApiBoundingBox(bounding_box.topleft.to_tuple(), bounding_box.size.x, bounding_box.size.y, bounding_box.size.z)
            if bounding_box is not None
            else None,
        )

        response = client.tasks_create(task_parameters)

        return cls._handle_task_creation_result(response)

    @classmethod
    def _from_dict(cls, response_dict: Dict) -> "Task":
        from ..client._generated.models.task_info_response_200 import (
            TaskInfoResponse200,
        )

        return cls._from_generated_response(
            TaskInfoResponse200.from_dict(response_dict)
        )

    @classmethod
    def _from_api_task(cls, api_task: ApiTask) -> "Task":
        return cls(
            api_task.id,
            api_task.project_id,
            api_task.data_set,
            TaskStatus(
                api_task.status.pending,
                api_task.status.active,
                api_task.status.finished,
            ),
            TaskType._from_api_task_type(api_task.type),
        )

    @classmethod
    def _from_generated_response(
        cls,
        response: Union["TaskInfoResponse200", "TaskInfosByProjectIdResponse200Item"],
    ) -> "Task":
        return cls(
            response.id,
            response.project_id,
            response.data_set,
            TaskStatus(
                response.status.pending,
                response.status.active,
                response.status.finished,
            ),
            TaskType._from_generated_response(response.type),
        )

    def get_annotation_infos(self) -> List[AnnotationInfo]:
        """Returns AnnotationInfo objects describing all task instances that have been started by annotators for this task"""
        client = _get_api_client(enforce_auth=True)
        api_annotations = client.annotation_infos_by_task(self, task.id)
        return [AnnotationInfo._from_api_annotation(a) for a in api_annotations]

    def get_project(self) -> Project:
        """Returns the project this task belongs to"""
        return Project.get_by_id(self.project_id)


    @classmethod
    def _handle_task_creation_result(cls, result: ApiTaskCreationResult) -> List["Task"]:
        if len(result.warnings) > 0:
            logger.warning(
                f"There were {len(result.warnings)} warnings during task creation:"
            )
        for warning in result.warnings:
            logger.warning(warning)

        successes = []
        errors = []
        for t in result.tasks:
            if t.success is not None:
                successes.append(t.success)
            if t.error is not None:
                errors.append(t.error)
        if len(errors) > 0:
            logger.error(f"{len(errors)} tasks could not be created:")
            for error in errors:
                logger.error(error)
        if len(successes) > 0:
            logger.info(f"{len(successes)} tasks were successfully created.")
        return [cls._from_api_task(t) for t in successes]
