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
from webknossos.geometry import BoundingBox

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
    task_id: str
    project_id: str
    dataset_name: str
    status: TaskStatus

    @classmethod
    def get_by_id(cls, id: str) -> "Task":  # pylint: disable=redefined-builtin
        client = _get_generated_client()
        response = task_info.sync(id=id, client=client)
        assert response is not None
        return cls._from_generated_response(response)

    @classmethod
    def create_from_annotations(
        cls,
        task_type_id: str,
        needed_experience_domain: str,
        needed_experience_value: int,
        instances: int,
        project_name: str,
        script_id: Optional[str],
        bounding_box: Optional[BoundingBox],
        base_annotations: List[Annotation],
    ) -> List["Task"]:
        assert (
            len(base_annotations) > 0
        ), "Must supply at least one base annotation to create tasks"
        client = _get_generated_client()
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
            f"{a.name}.zip": (f"{a.name}.zip", a.binary()) for a in base_annotations
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
        ), f"Failed to create tasks from files: {response.status_code}: {response.content.decode('utf-8')}"
        result = json.loads(response.content)
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
        client = _get_generated_client()
        response = annotation_infos_by_task_id.sync(id=self.task_id, client=client)
        assert response is not None
        return [AnnotationInfo._from_generated_response(a) for a in response]

    def get_project(self) -> Project:
        return Project.get_by_id(self.project_id)
