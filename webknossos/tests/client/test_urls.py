import pytest

import webknossos as wk
from webknossos.annotation.annotation import RemoteAnnotation

ANNOTATION_ID = "6114d9410100009f0096c640"


@pytest.mark.parametrize(
    "url",
    [
        f"https://example.org/annotations/{ANNOTATION_ID}",
        f"https://example.org/annotations/Explorational/{ANNOTATION_ID}",
        f"https://example.org/annotations/{ANNOTATION_ID}#9,9,9,0,1.3",
        f"https://example.org/annotations/Task/{ANNOTATION_ID}?token=abc#1,2,3",
    ],
)
def test_annotation_id_from_url(url: str) -> None:
    assert wk.Annotation.id_from_url(url) == ANNOTATION_ID


def test_annotation_id_from_id_is_unchanged() -> None:
    assert wk.Annotation.id_from_url(ANNOTATION_ID) == ANNOTATION_ID


@pytest.mark.parametrize(
    "url",
    [
        "https://example.org/annotations/",
        "https://example.org/annotations/#1,2,3",
        f"https://example.org/annotations/{ANNOTATION_ID}xyz",
        f"https://example.org/annotations/{ANNOTATION_ID}/",
        f"https://example.org/annotations/{ANNOTATION_ID}/view",
        f"https://example.org/annotations/Unknown/{ANNOTATION_ID}",
        f"example.org/annotations/{ANNOTATION_ID}",
    ],
)
def test_malformed_annotation_urls_are_not_matched(url: str) -> None:
    assert wk.Annotation.id_from_url(url) == url


def test_remote_annotation_generate_url_uses_its_own_context() -> None:
    with wk.webknossos_context(url="https://example.org"):
        annotation = RemoteAnnotation(
            ANNOTATION_ID,
            "my_org",
            skeleton=wk.Skeleton(dataset_name="ds", voxel_size=(1, 1, 1)),
            owner_name="me",
        )
    with wk.webknossos_context(url="https://other.example.com"):
        assert annotation.url == f"https://example.org/annotations/{ANNOTATION_ID}"
        assert (
            annotation.generate_url(position=(1, 2, 3))
            == f"https://example.org/annotations/{ANNOTATION_ID}#1,2,3,0,1"
        )
