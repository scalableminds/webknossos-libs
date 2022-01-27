from typing import List, Optional

import attr

from webknossos.administration import Project
from webknossos.annotation import Annotation, AnnotationInfo
from webknossos.client.context import _get_context
from webknossos.geometry import BoundingBox


@attr.frozen
class Task:
    task_id: str
    project_id: str
    dataset_name: str

    @classmethod
    def get_by_id(cls, id: str) -> "Task":  # pylint: disable=redefined-builtin
        from webknossos.client._get_task_info import get_task_info

        return get_task_info(id)

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

    def get_annotation_infos(self) -> List[AnnotationInfo]:
        return []

    def get_project(self) -> Project:
        return Project.get_by_id(self.project_id)
