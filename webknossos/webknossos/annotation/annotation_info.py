from typing import TYPE_CHECKING, Union

import attr

from webknossos.annotation.annotation import Annotation, AnnotationType

if TYPE_CHECKING:
    from webknossos.client._generated.models.annotation_info_response_200 import (
        AnnotationInfoResponse200,
    )
    from webknossos.client._generated.models.annotation_infos_by_task_id_response_200_item import (
        AnnotationInfosByTaskIdResponse200Item,
    )


@attr.frozen
class AnnotationInfo:
    id: str
    user_id: str
    name: str
    description: str
    type: AnnotationType

    def download_annotation(self) -> Annotation:
        from webknossos.client._download_annotation import download_annotation

        return download_annotation(self.type, self.id)

    @classmethod
    def _from_generated_response(
        cls,
        response: Union[
            "AnnotationInfoResponse200", "AnnotationInfosByTaskIdResponse200Item"
        ],
    ) -> "AnnotationInfo":
        return AnnotationInfo(
            id=response.id,
            user_id=response.user.id,
            name=response.name,
            description=response.description,
            type=AnnotationType(response.typ),
        )
