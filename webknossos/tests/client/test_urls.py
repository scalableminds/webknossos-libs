import pytest

import webknossos as wk


def test_annotation_url_of_an_id() -> None:
    assert (
        wk.annotation_url(
            "6114d9410100009f0096c640",
            position=(1, 2, 3),
            webknossos_url="https://example.org/",
        )
        == "https://example.org/annotations/6114d9410100009f0096c640#1,2,3,0,1"
    )


@pytest.mark.parametrize(
    "url",
    [
        "https://example.org/annotations/6114d9410100009f0096c640",
        "https://example.org/annotations/Explorational/6114d9410100009f0096c640",
        "https://example.org/annotations/6114d9410100009f0096c640#9,9,9,0,1.3",
    ],
)
def test_annotation_url_of_a_url_keeps_its_server(url: str) -> None:
    assert (
        wk.annotation_url(url, position=wk.Vec3Int(4, 5, 6))
        == "https://example.org/annotations/6114d9410100009f0096c640#4,5,6,0,1"
    )
    assert wk.annotation_id_from_url(url) == "6114d9410100009f0096c640"


def test_annotation_id_from_url_keeps_an_id() -> None:
    assert wk.annotation_id_from_url("6114d9410100009f0096c640") == (
        "6114d9410100009f0096c640"
    )


def test_urls_default_to_the_context() -> None:
    with wk.webknossos_context(url="https://context.example.org", token=None):
        assert wk.annotation_url("abc") == (
            "https://context.example.org/annotations/abc"
        )
        assert wk.dataset_url("68a1", position=(1, 2, 3)) == (
            "https://context.example.org/datasets/68a1/view#1,2,3,0,1"
        )
