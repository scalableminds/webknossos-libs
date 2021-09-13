import logging
from typing import Any, Dict, List, Optional, Tuple

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

logger = logging.getLogger(__name__)


# def globalize_tree_ids(group_dict: Dict[str, List[nx.Graph]]):
#     """
#     A utility to in-place re-assign new and globally unqiue IDs to all LegacyTree objects. Starts with ID 1

#     Arguments:
#         group_dict (Dict[str, List[nx.Graph]]): A mapping of group names to a list of trees as NetworkX graph objects
#     """

#     current_id = 1
#     for tree_group in group_dict.values():
#         for tree in tree_group:
#             tree["id"] = current_id
#             current_id += 1


# def globalize_node_ids(group_dict: Dict[str, List[nx.Graph]]):
#     """
#     A utility to in-place re-assign new and globally unqiue IDs to all Node objects. Edges are updated accordingly. Starts with ID 1.

#     Arguments:
#         group_dict (Dict[str, List[nx.Graph]]): A mapping of group names to a list of trees as NetworkX graph objects
#     """

#     current_id = 1
#     for tree_group in group_dict.values():
#         for tree_index in range(len(tree_group)):
#             tree = tree_group[tree_index]
#             new_tree = nx.Graph(**tree.nx_graph)
#             edge_mapping_dict = {}
#             for old_id in tree.nodes:
#                 tree.nodes[old_id]["id"] = current_id
#                 edge_mapping_dict[old_id] = current_id
#                 new_tree.add_node(current_id, **tree.nodes[old_id])

#                 current_id += 1
#             new_edges = []
#             for edge in tree.edges:
#                 new_edges.append(
#                     (edge_mapping_dict[edge[0]], edge_mapping_dict[edge[1]])
#                 )

#             new_tree.add_edges_from(new_edges)
#             tree_group[tree_index] = new_tree
