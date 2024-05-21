from typing import Optional

import attr

from ..client.api_client.models import ApiAnnotation
from ..utils import warn_deprecated
from .annotation import Annotation, AnnotationState, AnnotationType


@attr.frozen
class AnnotationInfo:
    """Data class containing information about a WEBKNOSSOS annotation"""

    id: str
    owner_id: Optional[str]
    name: str
    description: str
    type: AnnotationType
    state: AnnotationState
    duration_in_seconds: Optional[float]
    modified: Optional[int]

    def download_annotation(self) -> Annotation:
        """Downloads and returns the annotation that is described by this AnnotationInfo object"""
        return Annotation.download(self.id)

    @classmethod
    def _from_api_annotation(cls, api_annotation: ApiAnnotation) -> "AnnotationInfo":
        owner_id = api_annotation.owner.id if api_annotation.owner is not None else None
        return AnnotationInfo(
            id=api_annotation.id,
            owner_id=owner_id,
            name=api_annotation.name,
            description=api_annotation.description,
            type=AnnotationType(api_annotation.typ),
            state=AnnotationState(api_annotation.state),
            duration_in_seconds=(api_annotation.tracing_time / 1000)
            if api_annotation.tracing_time is not None
            else None,
            modified=api_annotation.modified,
        )

    @property
    def user_id(self) -> Optional[str]:
        warn_deprecated("user_id", "owner_id")
        return self.owner_id
