from webknossos.skeleton.entities import NML, Group, Node, WkGraph


def open_nml(file_path: str) -> "NML":
    return NML.from_path(file_path)
