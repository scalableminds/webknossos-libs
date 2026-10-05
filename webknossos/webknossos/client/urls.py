"""Links to annotations and datasets in WEBKNOSSOS, without contacting the server."""

import re

from ..geometry import Vec3Int, Vec3IntLike
from .context import _get_context


def _view(position: Vec3IntLike | None) -> str:
    if position is None:
        return ""
    return "#{},{},{},0,1".format(*Vec3Int(position).to_list())


def annotation_id_from_url(annotation_id_or_url: str) -> str:
    """Returns the id of an annotation URL, or the argument if it is no annotation URL.

    Examples:
        ```python
        annotation_id_from_url("https://webknossos.org/annotations/6114d9410100009f0096c640")
        # "6114d9410100009f0096c640"
        ```
    """
    from ..annotation.annotation import _ANNOTATION_URL_REGEX

    match = re.match(_ANNOTATION_URL_REGEX, annotation_id_or_url)
    return annotation_id_or_url if match is None else match.group("annotation_id")


def annotation_url(
    annotation_id_or_url: str,
    *,
    position: Vec3IntLike | None = None,
    webknossos_url: str | None = None,
) -> str:
    """Returns the URL of an annotation, which opens it at `position` (in mag 1) if given.

    Args:
        annotation_id_or_url: An annotation id, or an annotation URL whose WEBKNOSSOS
            server is used.
        position: The position to open the annotation at.
        webknossos_url: The WEBKNOSSOS server of an annotation id. Defaults to the URL of
            the current context.
    """
    from ..annotation.annotation import _ANNOTATION_URL_REGEX

    match = re.match(_ANNOTATION_URL_REGEX, annotation_id_or_url)
    if match is not None:
        webknossos_url = match.group("webknossos_url")
        annotation_id = match.group("annotation_id")
    else:
        annotation_id = annotation_id_or_url
    if webknossos_url is None:
        webknossos_url = _get_context().url
    return f"{webknossos_url.rstrip('/')}/annotations/{annotation_id}{_view(position)}"


def dataset_url(
    dataset_id: str,
    *,
    position: Vec3IntLike | None = None,
    webknossos_url: str | None = None,
) -> str:
    """Returns the URL that views a dataset, at `position` (in mag 1) if given.

    Args:
        dataset_id: The id of the dataset.
        position: The position to view the dataset at.
        webknossos_url: The WEBKNOSSOS server. Defaults to the URL of the current context.
    """
    if webknossos_url is None:
        webknossos_url = _get_context().url
    return f"{webknossos_url.rstrip('/')}/datasets/{dataset_id}/view{_view(position)}"
