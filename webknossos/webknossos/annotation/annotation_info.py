import attr

from ..client.api_client.models import ApiAnnotation
from .annotation import Annotation, AnnotationState, AnnotationType


@attr.frozen
class AnnotationInfo:
    """Data class containing information about a WEBKNOSSOS annotation"""

    id: str
    owner_id: str | None
    name: str
    description: str
    type: AnnotationType
    state: AnnotationState
    duration_in_seconds: float | None
    modified: int | None

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

    @classmethod
    def get_remote_annotations(
        cls, is_finished: bool | None = False, owner: str | None = None
    ) -> list["AnnotationInfo"]:
        """Returns a list of AnnotationInfo objects for all annotations that belong to the current user (if owner is None).
        If owner is not None, only annotations of the specified owner are returned."""
        from ..client.context import _get_api_client

        client = _get_api_client(True)
        if owner is None:
            owner = client.user_current().id

        return [
            cls._from_api_annotation(api_annotation)
            for api_annotation in client.annotation_list(is_finished)
            if api_annotation.owner.id == owner
        ]
