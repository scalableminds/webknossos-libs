from typing import TYPE_CHECKING, Optional, Union

import attr

from ..client._generated.types import Unset
from ..utils import warn_deprecated
from .annotation import Annotation, AnnotationState, AnnotationType

if TYPE_CHECKING:
    from ..client._generated.models.annotation_info_response_200 import (
        AnnotationInfoResponse200,
    )
    from ..client._generated.models.annotation_infos_by_task_id_response_200_item import (
        AnnotationInfosByTaskIdResponse200Item,
    )


@attr.frozen
class AnnotationInfo:
    """Data class containing information about a WEBKNOSSOS annotation"""

    id: str
    owner_id: Optional[str]
    name: str
    description: str
    type: AnnotationType
    state: AnnotationState
    duration_in_seconds: Optional[int]
    modified: Optional[int]

    def download_annotation(self) -> Annotation:
        """Downloads and returns the annotation that is discribed by this AnnotationInfo object"""
        return Annotation.download(self.id)

    @classmethod
    def _from_generated_response(
        cls,
        response: Union[
            "AnnotationInfoResponse200", "AnnotationInfosByTaskIdResponse200Item"
        ],
    ) -> "AnnotationInfo":
        maybe_owner = response.owner or response.user
        owner_id = None if isinstance(maybe_owner, Unset) else maybe_owner.id
        return AnnotationInfo(
            id=response.id,
            owner_id=owner_id,
            name=response.name,
            description=response.description,
            type=AnnotationType(response.typ),
            state=AnnotationState(response.state),
            duration_in_seconds=response.tracing_time // 1000
            if response.tracing_time is not None
            and not isinstance(response.tracing_time, Unset)
            else None,
            modified=response.modified,
        )

    @property
    def user_id(self) -> Optional[str]:
        warn_deprecated("user_id", "owner_id")
        return self.owner_id
