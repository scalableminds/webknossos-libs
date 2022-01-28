from typing import TYPE_CHECKING, List, Optional, Union

import attr

from webknossos.administration import Project
from webknossos.annotation import Annotation, AnnotationInfo
from webknossos.client._generated.api.default import (
    annotation_infos_by_task_id,
    task_info,
)
from webknossos.client.context import _get_context, _get_generated_client
from webknossos.geometry import BoundingBox

if TYPE_CHECKING:
    from webknossos.client._generated.models.task_info_response_200 import (
        TaskInfoResponse200,
    )
    from webknossos.client._generated.models.task_infos_by_project_id_response_200_item import (
        TaskInfosByProjectIdResponse200Item,
    )


@attr.frozen
class Task:
    task_id: str
    project_id: str
    dataset_name: str

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
    ) -> "Task":
        context = _get_context()
        token = context.required_token

    @classmethod
    def _from_generated_response(
        cls,
        response: Union["TaskInfoResponse200", "TaskInfosByProjectIdResponse200Item"],
    ) -> "Task":
        return cls(response.id, response.project_name, response.data_set)

    def get_annotation_infos(self) -> List[AnnotationInfo]:
        client = _get_generated_client()
        response = annotation_infos_by_task_id.sync(id=self.task_id, client=client)
        assert response is not None
        return [AnnotationInfo._from_generated_response(a) for a in response]

    def get_project(self) -> Project:
        return Project.get_by_id(self.project_id)
