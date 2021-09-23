import itertools
from typing import TYPE_CHECKING, List

import webknossos.skeleton.nml as wknml

if TYPE_CHECKING:
    from webknossos.skeleton import Graph, Group, Skeleton


def to_skeleton(nml: wknml.NML) -> "Skeleton":
    from webknossos.skeleton import Skeleton

    skeleton = Skeleton(
        name=nml.parameters.name,
        scale=nml.parameters.scale,
        offset=nml.parameters.offset,
        time=nml.parameters.time,
        edit_position=nml.parameters.editPosition,
        edit_rotation=nml.parameters.editRotation,
        zoom_level=nml.parameters.zoomLevel,
        task_bounding_box=nml.parameters.taskBoundingBox,
        user_bounding_boxes=nml.parameters.userBoundingBoxes,
    )

    groups_by_id = {}

    def visit_groups(nml_groups: List[wknml.Group], current_group: "Group") -> None:
        for nml_group in nml_groups:
            sub_group = current_group.add_group(
                name=nml_group.name, _enforced_id=nml_group.id
            )
            groups_by_id[sub_group.id] = sub_group
            visit_groups(nml_group.children, sub_group)

    visit_groups(nml.groups, skeleton)
    for nml_tree in nml.trees:
        if nml_tree.groupId is None:
            new_graph = skeleton.add_graph(
                nml_tree.name, _enforced_id=nml_tree.id, color=nml_tree.color
            )
        else:
            new_graph = groups_by_id[nml_tree.groupId].add_graph(
                nml_tree.name, _enforced_id=nml_tree.id, color=nml_tree.color
            )
        _nml_tree_to_graph(new_graph, nml_tree)

    for comment in nml.comments:
        skeleton.get_node_by_id(comment.node).comment = comment.content

    for branchpoint in nml.branchpoints:
        node = skeleton.get_node_by_id(branchpoint.id)
        node.is_branchpoint = True
        if branchpoint.time != 0:
            node.branchpoint_time = branchpoint.time

    max_id = max(skeleton.get_max_graph_id(), skeleton.get_max_node_id())
    skeleton.element_id_generator = itertools.count(max_id + 1)

    return skeleton


def _nml_tree_to_graph(
    new_graph: "Graph",
    nml_tree: wknml.Tree,
) -> None:
    optional_attribute_list = [
        "rotation",
        "inVp",
        "inMag",
        "bitDepth",
        "interpolation",
        "time",
    ]

    for nml_node in nml_tree.nodes:
        node_id = nml_node.id
        current_node = new_graph.add_node(
            position=nml_node.position,
            _enforced_id=node_id,
            radius=nml_node.radius,
        )

        for optional_attribute in optional_attribute_list:
            if getattr(nml_node, optional_attribute) is not None:
                setattr(
                    current_node,
                    optional_attribute,
                    getattr(nml_node, optional_attribute),
                )

    for edge in nml_tree.edges:
        new_graph.add_edge(edge.source, edge.target)
