from io import BytesIO

from webknossos.annotation.annotation import Annotation, AnnotationType
from webknossos.client.context import get_generated_client
from webknossos.client.generated.api.default import annotation_download

COMPOUND_ANNOTATION_TYPES = [
    AnnotationType.COMPOUND_PROJECT,
    AnnotationType.COMPOUND_TASK,
    AnnotationType.COMPOUND_TASK_TYPE,
]


def download_annotation(
    type: AnnotationType, id: str  # pylint: disable=redefined-builtin
) -> Annotation:
    assert (
        type not in COMPOUND_ANNOTATION_TYPES
    ), f"Currently compund annotation types are not supported, got {type}"
    client = get_generated_client()
    response = annotation_download.sync_detailed(typ=type.value, id=id, client=client)
    assert response.status_code == 200
    return Annotation(BytesIO(response.content))
