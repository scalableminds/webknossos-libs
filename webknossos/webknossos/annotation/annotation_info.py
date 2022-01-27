import attr

from webknossos.annotation.annotation import Annotation, AnnotationType


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
