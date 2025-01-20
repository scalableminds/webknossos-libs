import logging
from typing import List, Optional, Union

import attr

from ..annotation import Annotation, AnnotationInfo
from ..client.api_client.models import (
    ApiBoundingBox,
    ApiExperience,
    ApiNmlTaskParameters,
    ApiTask,
    ApiTaskCreationResult,
    ApiTaskParameters,
    ApiTaskType,
)
from ..client.context import _get_api_client, _get_context
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

        client = _get_api_client(enforce_auth=True)
        nml_task_parameters = ApiNmlTaskParameters(
            task_type_id=task_type_id,
            needed_experience=ApiExperience(
                domain=needed_experience_domain,
                value=needed_experience_value,
            ),
            pending_instances=instances,
            project_name=project_name,
            script_id=script_id,
            bounding_box=ApiBoundingBox(
                bounding_box.topleft.to_tuple(),
                bounding_box.size.x,
                bounding_box.size.y,
                bounding_box.size.z,
            )
            if bounding_box is not None
            else None,
        )
        annotation_files = [
            (f"{a.name}.zip", a._binary_zip()) for a in base_annotations
        ]
        result = client.tasks_create_from_files(nml_task_parameters, annotation_files)

        return cls._handle_task_creation_result(result)

    @classmethod
    def create(
        cls,
        task_type_id: str,
        project_name: str,
        needed_experience_domain: str,
        needed_experience_value: int,
        starting_position: Vec3Int,
        dataset_name: Optional[Union[str, RemoteDataset]] = None,
        starting_rotation: Vec3Int = Vec3Int(0, 0, 0),
        instances: int = 1,
        dataset_id: Optional[Union[str, RemoteDataset]] = None,
        script_id: Optional[str] = None,
        bounding_box: Optional[BoundingBox] = None,
    ) -> List["Task"]:
        """Submits tasks in WEBKNOSSOS based on a dataset, starting position + rotation, and returns the Task objects"""

        client = _get_api_client(enforce_auth=True)
        assert (
            dataset_id is not None or dataset_name is not None
        ), "Please provide a dataset_id to create a task."
        if dataset_id is not None:
            if isinstance(dataset_id, RemoteDataset):
                dataset_id = dataset_id._dataset_id
        else:
            assert dataset_name is not None
            warn_deprecated("dataset_name", "dataset_id")
            if isinstance(dataset_name, RemoteDataset):
                dataset_id = dataset_name._dataset_id
            else:
                context = _get_context()
                dataset_id = client.dataset_id_from_name(
                    dataset_name, context.organization_id
                )

        task_parameters = ApiTaskParameters(
            task_type_id=task_type_id,
            needed_experience=ApiExperience(
                domain=needed_experience_domain,
                value=needed_experience_value,
            ),
            pending_instances=instances,
            project_name=project_name,
            script_id=script_id,
            dataset_id=dataset_id,
            edit_position=starting_position.to_tuple(),
            edit_rotation=starting_rotation.to_tuple(),
            bounding_box=ApiBoundingBox(
                bounding_box.topleft.to_tuple(),
                bounding_box.size.x,
                bounding_box.size.y,
                bounding_box.size.z,
            )
            if bounding_box is not None
            else None,
        )

        response = client.tasks_create([task_parameters])

        return cls._handle_task_creation_result(response)

    @classmethod
    def _from_api_task(cls, api_task: ApiTask) -> "Task":
        return cls(
            api_task.id,
            api_task.project_id,
            api_task.dataset_name,
            TaskStatus(
                api_task.status.pending,
                api_task.status.active,
                api_task.status.finished,
            ),
            TaskType._from_api_task_type(api_task.type),
        )

    def get_annotation_infos(self) -> List[AnnotationInfo]:
        """Returns AnnotationInfo objects describing all task instances that have been started by annotators for this task"""
        client = _get_api_client(enforce_auth=True)
        api_annotations = client.annotation_infos_by_task(self.task_id)
        return [AnnotationInfo._from_api_annotation(a) for a in api_annotations]

    def get_project(self) -> Project:
        """Returns the project this task belongs to"""
        return Project.get_by_id(self.project_id)

    @classmethod
    def _handle_task_creation_result(
        cls, result: ApiTaskCreationResult
    ) -> List["Task"]:
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
