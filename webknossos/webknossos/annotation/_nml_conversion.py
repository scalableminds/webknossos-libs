import colorsys
import itertools
import warnings
from typing import TYPE_CHECKING, List, Tuple

import networkx as nx
import numpy as np

import webknossos._nml as wknml
from webknossos.geometry import Vec3Int

if TYPE_CHECKING:
    from webknossos.annotation import Annotation
    from webknossos.skeleton import Graph, Group, Skeleton


def nml_to_skeleton(nml: wknml.Nml) -> "Skeleton":
    from webknossos.skeleton import Skeleton

    skeleton = Skeleton(
        dataset_name=nml.parameters.name,
        scale=nml.parameters.scale,
        organization_id=nml.parameters.organization,
        description=nml.parameters.description,
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
    skeleton._element_id_generator = itertools.count(max_id + 1)

    if nml.parameters.offset is not None and not all(
        i == 0 for i in nml.parameters.offset
    ):
        warnings.warn(
            f"The offset parameter is not supported. The offset of {nml.parameters.offset} is ignored and will be omitted."
        )

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
            position=Vec3Int.from_vec3_float(nml_node.position),
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


def _random_color_rgba() -> Tuple[float, float, float, float]:
    """
    A utility to generate a new random RGBA color.
    """
    # https://stackoverflow.com/a/43437435/783758

    h, s, l = (
        np.random.random(),
        0.5 + np.random.random() / 2.0,
        0.4 + np.random.random() / 5.0,
    )
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return (r, g, b, 1.0)


def annotation_to_nml(  # pylint: disable=dangerous-default-value
    annotation: "Annotation",
) -> wknml.Nml:

    nmlParameters = wknml.Parameters(
        name=annotation.dataset_name,
        scale=annotation.scale,
        description=annotation.description,
        organization=annotation.organization_id,
        time=annotation.time,
        editPosition=annotation.edit_position,
        editRotation=annotation.edit_rotation,
        zoomLevel=annotation.zoom_level,
        taskBoundingBox=annotation.task_bounding_box,
        userBoundingBoxes=annotation.user_bounding_boxes,
    )

    comments = [
        wknml.Comment(node.id, node.comment)
        for graph in annotation.skeleton.flattened_graphs()
        for node in graph.nodes
        if node.comment is not None
    ]

    branchpoints = [
        wknml.Branchpoint(node.id, node.time)
        for graph in annotation.skeleton.flattened_graphs()
        for node in graph.nodes
        if node.is_branchpoint
    ]

    graphs = []

    for graph in sorted(annotation.skeleton.flattened_graphs(), key=lambda g: g.id):
        nodes, edges = _extract_nodes_and_edges_from_graph(graph)
        color = graph.color or _random_color_rgba()
        name = graph.name or f"tree{graph.id}"

        graphs.append(
            wknml.Tree(
                nodes=nodes,
                edges=edges,
                id=graph.id,
                name=name,
                groupId=graph.group.id if graph.group != annotation.skeleton else None,
                color=color,
            )
        )

    groups = annotation.skeleton.as_nml_group().children

    volumes = []
    for volume in annotation._volume_layers:
        volumes.append(
            wknml.Volume(
                id=volume.id,
                location=volume._default_zip_name(),
                fallback_layer=volume.fallback_layer_name,
                name=volume.name,
            )
        )

    meta = [
        wknml.Meta(name=key, content=value)
        for key, value in annotation.metadata.items()
    ]
    if annotation.username is not None:
        meta.append(wknml.Meta(name="username", content=annotation.username))
    if annotation.annotation_id is not None:
        meta.append(wknml.Meta(name="annotationId", content=annotation.annotation_id))

    nml = wknml.Nml(
        meta=meta,
        parameters=nmlParameters,
        trees=graphs,
        branchpoints=branchpoints,
        comments=comments,
        groups=groups,
        volumes=volumes,
    )

    return nml


def _extract_nodes_and_edges_from_graph(
    graph: nx.Graph,
) -> Tuple[List[wknml.Node], List[wknml.Edge]]:
    """
    A utility to convert a single [NetworkX graph object](https://networkx.org/) into a list of `NmlNode` objects and `Edge` objects.

    Return
        Tuple[List[NmlNode], List[Edge]]: A tuple containing both all nodes and all edges
    """

    node_nml = [
        wknml.Node(
            id=node.id,
            position=node.position,
            radius=node.radius,
            rotation=node.rotation,
            inVp=node.inVp,
            inMag=node.inMag,
            bitDepth=node.bitDepth,
            interpolation=node.interpolation,
            time=node.time,
        )
        for node in graph.nodes
    ]

    edge_nml = [
        wknml.Edge(source=source.id, target=target.id) for source, target in graph.edges
    ]

    return node_nml, edge_nml
