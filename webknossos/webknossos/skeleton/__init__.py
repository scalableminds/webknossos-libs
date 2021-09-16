from webknossos.skeleton.entities import Skeleton, Group, Node, WkGraph


def open_nml(file_path: str) -> Skeleton:
    return Skeleton.from_path(file_path)
