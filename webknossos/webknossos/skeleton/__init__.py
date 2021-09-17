from webknossos.skeleton.entities import Graph, Group, Node, Skeleton


def open_nml(file_path: str) -> Skeleton:
    return Skeleton.from_path(file_path)
