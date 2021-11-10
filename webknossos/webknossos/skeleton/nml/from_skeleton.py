import colorsys
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from webknossos.skeleton.nml import NML
from webknossos.skeleton.nml import Branchpoint as NmlBranchpoint
from webknossos.skeleton.nml import Comment as NmlComment
from webknossos.skeleton.nml import Edge as NmlEdge
from webknossos.skeleton.nml import NMLParameters as NmlParameters
from webknossos.skeleton.nml import Node as NmlNode
from webknossos.skeleton.nml import Tree as NmlTree
from webknossos.skeleton.nml import Volume as NmlVolume
from webknossos.skeleton.nml import enforce_not_null

if TYPE_CHECKING:
    from webknossos.skeleton import Group


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


def from_skeleton(
    group: "Group",
    parameters: Optional[Dict[str, Any]] = None,
    volume_dict: Optional[Dict[str, Any]] = None,
) -> NML:
    """
    A utility to convert a [NetworkX graph object](https://networkx.org/) into wK NML skeleton annotation object. Accepts both a simple list of multiple skeletons/trees or a dictionary grouping skeleton inputs.

    Arguments:
        tree_dict (Union[List[nx.Graph], Dict[str, List[nx.Graph]]]): A list of wK tree-like structures as NetworkX graphs or a dictionary of group names and same lists of NetworkX tree objects.
        parameters (Dict[str, Any]): A dictionary representation of the skeleton annotation metadata. See `NmlParameters` for accepted attributes.
        volume (Optional[Dict[str, Any]] = None): A dictionary representation of a reference to a wK volume annotation. See `NmlVolume` object for attributes.

    Return:
        nml (NML): A wK NML skeleton annotation object
    """

    parameters = parameters or {}

    nmlParameters = NmlParameters(
        name=parameters.get("name", "dataset"),
        scale=parameters.get("scale", None),
        offset=parameters.get("offset", None),
        time=parameters.get("time", None),
        editPosition=parameters.get("editPosition", None),
        editRotation=parameters.get("editRotation", None),
        zoomLevel=parameters.get("zoomLevel", None),
        taskBoundingBox=parameters.get("taskBoundingBox", None),
        userBoundingBoxes=parameters.get("userBoundingBoxes", None),
    )

    comments = [
        NmlComment(node.id, node.comment)
        for graph in group.flattened_graphs()
        for node in graph.nodes
        if node.comment is not None
    ]

    branchpoints = [
        NmlBranchpoint(node.id, node.time)
        for graph in group.flattened_graphs()
        for node in graph.nodes
        if node.is_branchpoint
    ]

    graphs = []

    for graph in sorted(group.flattened_graphs(), key=lambda g: g.id):
        nodes, edges = _extract_nodes_and_edges_from_graph(graph)
        color = graph.color or _random_color_rgba()
        name = graph.name or f"tree{graph.id}"

        graphs.append(
            NmlTree(
                nodes=nodes,
                edges=edges,
                id=graph.id,
                name=name,
                groupId=graph.group.id if graph.group != group else None,
                color=color,
            )
        )

    volume = None
    if volume_dict is not None and "location" in volume_dict and "id" in volume_dict:
        volume = NmlVolume(
            id=int(enforce_not_null(volume_dict.get("id"))),
            location=enforce_not_null(volume_dict.get("location")),
            fallback_layer=volume_dict.get("fallback_layer"),
        )

    nml = NML(
        parameters=nmlParameters,
        trees=graphs,
        branchpoints=branchpoints,
        comments=comments,
        groups=group.as_nml_group().children,
        volume=volume,
    )

    return nml


def _extract_nodes_and_edges_from_graph(
    graph: nx.Graph,
) -> Tuple[List[NmlNode], List[NmlEdge]]:
    """
    A utility to convert a single [NetworkX graph object](https://networkx.org/) into a list of `NmlNode` objects and `Edge` objects.

    Return
        Tuple[List[NmlNode], List[Edge]]: A tuple containing both all nodes and all edges
    """

    node_nml = [
        NmlNode(
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
        NmlEdge(source=source.id, target=target.id) for source, target in graph.edges
    ]

    return node_nml, edge_nml
