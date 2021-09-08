from wknml import (
    NMLParameters,
    Group,
    Edge,
    Node,
    Tree,
    NML,
    Branchpoint,
    Comment,
    Volume,
)
import networkx as nx
import numpy as np

import logging
import colorsys
from typing import Optional, Text, Tuple, List, Dict, Union, Any
from copy import deepcopy


logger = logging.getLogger(__name__)


def random_color_rgba() -> Tuple[float, float, float, float]:
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


def discard_children_hierarchy(groups: List[Group]) -> List[Group]:
    """
    A utility to flatten the group structure. All sub-groups will become top-level items.
    """

    groups_without_hierarchy = []
    for group in groups:
        children = discard_children_hierarchy(group.children)
        groups_without_hierarchy.append(
            Group(id=group.id, name=group.name, children=[])
        )
        groups_without_hierarchy.extend(children)
    return groups_without_hierarchy


def globalize_tree_ids(group_dict: Dict[str, List[nx.Graph]]):
    """
    A utility to in-place re-assign new and globally unqiue IDs to all Tree objects. Starts with ID 1

    Arguments:
        group_dict (Dict[str, List[nx.Graph]]): A mapping of group names to a list of trees as NetworkX graph objects
    """

    current_id = 1
    for tree_group in group_dict.values():
        for tree in tree_group:
            tree.graph["id"] = current_id
            current_id += 1


def globalize_node_ids(group_dict: Dict[str, List[nx.Graph]]):
    """
    A utility to in-place re-assign new and globally unqiue IDs to all Node objects. Edges are updated accordingly. Starts with ID 1.

    Arguments:
        group_dict (Dict[str, List[nx.Graph]]): A mapping of group names to a list of trees as NetworkX graph objects
    """

    current_id = 1
    for tree_group in group_dict.values():
        for tree_index in range(len(tree_group)):
            tree = tree_group[tree_index]
            new_tree = nx.Graph(**tree.graph)
            edge_mapping_dict = {}
            for old_id in tree.nodes:
                tree.nodes[old_id]["id"] = current_id
                edge_mapping_dict[old_id] = current_id
                new_tree.add_node(current_id, **tree.nodes[old_id])

                current_id += 1
            new_edges = []
            for edge in tree.edges:
                new_edges.append(
                    (edge_mapping_dict[edge[0]], edge_mapping_dict[edge[1]])
                )

            new_tree.add_edges_from(new_edges)
            tree_group[tree_index] = new_tree


def generate_nml(
    tree_dict: Union[List[nx.Graph], Dict[str, List[nx.Graph]]],
    parameters: Dict[str, Any] = {},
    globalize_ids: bool = True,
    volume: Optional[Dict[str, Any]] = None,
) -> NML:
    """
    A utility to convert a [NetworkX graph object](https://networkx.org/) into wK NML skeleton annotation object. Accepts both a simple list of multiple skeletons/trees or a dictionary grouping skeleton inputs.

    Arguments:
        tree_dict (Union[List[nx.Graph], Dict[str, List[nx.Graph]]]): A list of wK tree-like structures as NetworkX graphs or a dictionary of group names and same lists of NetworkX tree objects.
        parameters (Dict[str, Any]): A dictionary representation of the skeleton annotation metadata. See `NMLParameters` for accepted attributes.
        globalize_ids (bool = True): An option to re-assign new, globally unique IDs to all skeletons. Default: `True`
        volume (Optional[Dict[str, Any]] = None): A dictionary representation of a reference to a wK volume annotation. See `Volume` object for attributes.

    Return:
        nml (NML): A wK NML skeleton annotation object
    """
    no_group_provided = False
    if not isinstance(tree_dict, dict):
        tree_dict = {"main_group": tree_dict}
        no_group_provided = True

    if globalize_ids:
        globalize_tree_ids(tree_dict)
        globalize_node_ids(tree_dict)

    nmlParameters = NMLParameters(
        name=parameters.get("name", "dataset"),
        scale=parameters.get("scale", [11.24, 11.24, 25]),
        offset=parameters.get("offset", None),
        time=parameters.get("time", None),
        editPosition=parameters.get("editPosition", None),
        editRotation=parameters.get("editRotation", None),
        zoomLevel=parameters.get("zoomLevel", None),
        taskBoundingBox=parameters.get("taskBoundingBox", None),
        userBoundingBoxes=parameters.get("userBoundingBoxes", None),
    )

    comments = [
        Comment(node, tree.nodes[node]["comment"])
        for group in tree_dict.values()
        for tree in group
        for node in tree.nodes
        if "comment" in tree.nodes[node]
    ]

    branchpoints = [
        Branchpoint(tree.nodes[node]["id"], 0)
        for group in tree_dict.values()
        for tree in group
        for node in tree.nodes
        if "branchpoint" in tree.nodes[node]
    ]

    if no_group_provided:
        groups = []
    else:
        groups = [
            Group(id=group_id, name=group_name, children=[])
            for group_id, group_name in enumerate(tree_dict, 1)
        ]

    trees = []

    for group_id, group_name in enumerate(tree_dict, 1):
        for tree in tree_dict[group_name]:
            nodes, edges = extract_nodes_and_edges_from_graph(tree)
            color = tree.graph.get("color", random_color_rgba())
            name = tree.graph.get("name", f"tree{tree.graph['id']}")

            trees.append(
                Tree(
                    nodes=nodes,
                    edges=edges,
                    id=tree.graph["id"],
                    name=name,
                    groupId=None if no_group_provided else group_id,
                    color=color,
                )
            )

    if volume is not None and "location" in volume and "id" in volume:
        volume = Volume(
            id=volume.get("id"),
            location=volume.get("location"),
            fallback_layer=volume.get("fallback_layer", None),
        )

    nml = NML(
        parameters=nmlParameters,
        trees=trees,
        branchpoints=branchpoints,
        comments=comments,
        groups=groups,
        volume=volume,
    )

    return nml


def generate_graph(nml: NML) -> Tuple[Dict[str, List[nx.Graph]], Dict[Text, any]]:
    """
    A utility to convert a wK NML object into a [NetworkX graph object](https://networkx.org/). Skeletons/Trees are grouped by the provided groups in the NML file.

    Arguments:
        nml (NML): A wK NML skeleton annotation object

    Return:
        A tuple consisting of:
            1. A dictionary with group names as keys and lists of all respective NML trees as values
            2. A dictionary representation of the NML metadata parameters. See `NMLParameters` for attributes.
    """

    nml = deepcopy(nml)._replace(groups=discard_children_hierarchy(nml.groups))
    group_dict = {}
    for group in nml.groups:
        graphs_in_current_group = []
        for tree in nml.trees:
            if tree.groupId == group.id:
                graphs_in_current_group.append(nml_tree_to_graph(tree))
        group_dict[group.name] = graphs_in_current_group

    nml_parameters = nml.parameters
    parameter_dict = {}

    parameter_list = [
        "name",
        "scale",
        "offset",
        "time",
        "editPosition",
        "editRotation",
        "zoomLevel",
        "taskBoundingBox",
        "userBoundingBoxes",
    ]

    for parameter in parameter_list:
        if getattr(nml_parameters, parameter) is not None:
            parameter_dict[parameter] = getattr(nml_parameters, parameter)

    for comment in nml.comments:
        for group in group_dict.values():
            for tree in group:
                if comment.node in tree.nodes:
                    tree.nodes[comment.node]["comment"] = comment.content

    for branchpoint in nml.branchpoints:
        for group in group_dict.values():
            for tree in group:
                if branchpoint.id in tree.nodes:
                    tree.nodes[branchpoint.id]["branchpoint"] = branchpoint.time

    return group_dict, parameter_dict


def nml_tree_to_graph(tree: Tree) -> nx.Graph:
    """
    A utility to convert a single wK Tree object into a [NetworkX graph object](https://networkx.org/).
    """

    optional_attribute_list = [
        "rotation",
        "inVp",
        "inMag",
        "bitDepth",
        "interpolation",
        "time",
    ]

    graph = nx.Graph(id=tree.id, color=tree.color, name=tree.name, groupId=tree.groupId)
    for node in tree.nodes:
        node_id = node.id
        graph.add_node(node_id, id=node_id, radius=node.radius, position=node.position)

        for optional_attribute in optional_attribute_list:
            if getattr(node, optional_attribute) is not None:
                graph.nodes[node_id][optional_attribute] = getattr(
                    node, optional_attribute
                )

    graph.add_edges_from([(edge.source, edge.target) for edge in tree.edges])

    return graph


def extract_nodes_and_edges_from_graph(
    graph: nx.Graph,
) -> Tuple[List[Node], List[Edge]]:
    """
    A utility to convert a single [NetworkX graph object](https://networkx.org/) into a list of `Node` objects and `Edge` objects.

    Return
        Tuple[List[Node], List[Edge]]: A tuple containing both all nodes and all edges
    """

    node_nml = [
        Node(
            id=graph.nodes[node]["id"],
            position=graph.nodes[node]["position"],
            radius=graph.nodes[node].get("radius", 1.0),
            rotation=graph.nodes[node].get("rotation", None),
            inVp=graph.nodes[node].get("inVp", None),
            inMag=graph.nodes[node].get("inMag", None),
            bitDepth=graph.nodes[node].get("bitDepth", None),
            interpolation=graph.nodes[node].get("interpolation", None),
            time=graph.nodes[node].get("time", None),
        )
        for node in graph.nodes
    ]

    edge_nml = [Edge(source=edge[0], target=edge[1]) for edge in graph.edges]

    return node_nml, edge_nml
