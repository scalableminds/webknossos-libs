import colorsys
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from webknossos.skeleton.legacy import NML as LegacyNML
from webknossos.skeleton.legacy import Branchpoint as LegacyBranchpoint
from webknossos.skeleton.legacy import Comment as LegacyComment
from webknossos.skeleton.legacy import Edge as LegacyEdge
from webknossos.skeleton.legacy import NMLParameters as LegacyNMLParameters
from webknossos.skeleton.legacy import Node as LegacyNode
from webknossos.skeleton.legacy import Tree as LegacyTree
from webknossos.skeleton.legacy import Volume as LegacyVolume
from webknossos.skeleton.legacy import enforce_not_null_str

if TYPE_CHECKING:
    from webknossosk.skeleton import Group


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


class NMLExporter:
    @staticmethod
    def generate_nml(
        group: "Group",
        parameters: Dict[str, Any] = None,
        volume_dict: Optional[Dict[str, Any]] = None,
    ) -> LegacyNML:
        """
        A utility to convert a [NetworkX graph object](https://networkx.org/) into wK NML skeleton annotation object. Accepts both a simple list of multiple skeletons/trees or a dictionary grouping skeleton inputs.

        Arguments:
            tree_dict (Union[List[nx.Graph], Dict[str, List[nx.Graph]]]): A list of wK tree-like structures as NetworkX graphs or a dictionary of group names and same lists of NetworkX tree objects.
            parameters (Dict[str, Any]): A dictionary representation of the skeleton annotation metadata. See `LegacyNMLParameters` for accepted attributes.
            volume (Optional[Dict[str, Any]] = None): A dictionary representation of a reference to a wK volume annotation. See `LegacyVolume` object for attributes.

        Return:
            nml (LegacyNML): A wK LegacyNML skeleton annotation object
        """

        parameters = parameters or {}

        nmlParameters = LegacyNMLParameters(
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
            LegacyComment(node.id, node.comment)
            for graph in group.flattened_graphs()
            for _id, node in graph.nx_graph.nodes(data="obj")
            if node.comment is not None
        ]

        branchpoints = [
            LegacyBranchpoint(node.id, node.time)
            for graph in group.flattened_graphs()
            for _id, node in graph.nx_graph.nodes(data="obj")
            if node.is_branchpoint
        ]

        graphs = []

        for graph in sorted(group.flattened_graphs(), key=lambda g: g.id):

            nodes, edges = NMLExporter.extract_nodes_and_edges_from_graph(graph)
            color = graph.color or random_color_rgba()
            name = graph.name or f"tree{graph.id}"

            graphs.append(
                LegacyTree(
                    nodes=nodes,
                    edges=edges,
                    id=graph.id,
                    name=name,
                    groupId=graph.group_id if graph.group_id != group.id else None,
                    color=color,
                )
            )

        volume = None
        if (
            volume_dict is not None
            and "location" in volume_dict
            and "id" in volume_dict
        ):
            volume = LegacyVolume(
                id=int(enforce_not_null_str(volume_dict.get("id"))),
                location=enforce_not_null_str(volume_dict.get("location")),
                fallback_layer=volume_dict.get("fallback_layer"),
            )

        nml = LegacyNML(
            parameters=nmlParameters,
            trees=graphs,
            branchpoints=branchpoints,
            comments=comments,
            groups=group.as_legacy_group().children,
            volume=volume,
        )

        return nml

    @staticmethod
    def extract_nodes_and_edges_from_graph(
        graph: nx.Graph,
    ) -> Tuple[List[LegacyNode], List[LegacyEdge]]:
        """
        A utility to convert a single [NetworkX graph object](https://networkx.org/) into a list of `LegacyNode` objects and `Edge` objects.

        Return
            Tuple[List[LegacyNode], List[Edge]]: A tuple containing both all nodes and all edges
        """

        node_nml = [
            LegacyNode(
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
            for _id, node in graph.nx_graph.nodes(data="obj")
        ]

        edge_nml = [
            LegacyEdge(source=edge[0], target=edge[1]) for edge in graph.nx_graph.edges
        ]

        return node_nml, edge_nml
