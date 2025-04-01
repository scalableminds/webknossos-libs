import colorsys
import itertools
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

import webknossos._nml as wknml

from ..geometry import Vec3Int

if TYPE_CHECKING:
    from ..annotation import Annotation
    from ..skeleton import Group, Skeleton, Tree


def nml_to_skeleton(nml: wknml.Nml) -> "Skeleton":
    from ..skeleton import Skeleton

    skeleton = Skeleton(
        dataset_id=nml.parameters.dataset_id,
        dataset_name=nml.parameters.name,
        voxel_size=nml.parameters.scale,
        organization_id=nml.parameters.organization,
        description=nml.parameters.description,
    )

    groups_by_id = {}

    def visit_groups(nml_groups: list[wknml.Group], current_group: "Group") -> None:
        for nml_group in nml_groups:
            sub_group = current_group.add_group(
                name=nml_group.name, _enforced_id=nml_group.id
            )
            groups_by_id[sub_group.id] = sub_group
            visit_groups(nml_group.children, sub_group)

    visit_groups(nml.groups, skeleton)
    for nml_tree in nml.trees:
        if nml_tree.groupId is None:
            new_tree = skeleton.add_tree(
                nml_tree.name,
                _enforced_id=nml_tree.id,
                color=nml_tree.color,
                metadata={i.key: i.value for i in nml_tree.metadata},
            )
        else:
            new_tree = groups_by_id[nml_tree.groupId].add_tree(
                nml_tree.name,
                _enforced_id=nml_tree.id,
                color=nml_tree.color,
                metadata={i.key: i.value for i in nml_tree.metadata},
            )
        _nml_tree_to_wk_tree(new_tree, nml_tree)

    for comment in nml.comments:
        skeleton.get_node_by_id(comment.node).comment = comment.content

    for branchpoint in nml.branchpoints:
        node = skeleton.get_node_by_id(branchpoint.id)
        node.is_branchpoint = True
        if branchpoint.time != 0:
            node.branchpoint_time = branchpoint.time

    max_id = max(skeleton.get_max_tree_id(), skeleton.get_max_node_id())
    skeleton._element_id_generator = itertools.count(max_id + 1)

    if nml.parameters.offset is not None and not all(
        i == 0 for i in nml.parameters.offset
    ):
        warnings.warn(
            f"[INFO] The offset parameter is not supported. The offset of {nml.parameters.offset} is ignored and will be omitted."
        )

    return skeleton


def _dict_to_metadata_entry_list(
    metadata: dict[str, str | int | float | Sequence[str]],
) -> list[wknml.MetadataEntry]:
    metadata_entry_list = []
    if metadata is not None:
        for key, value in metadata.items():
            if isinstance(value, str):
                metadata_entry_list.append(
                    wknml.MetadataEntry(key=key, type="str", value=value)
                )
            elif isinstance(value, int) or isinstance(value, float):
                metadata_entry_list.append(
                    wknml.MetadataEntry(key=key, type="number", value=value)
                )
            elif isinstance(value, Sequence):
                metadata_entry_list.append(
                    wknml.MetadataEntry(key=key, type="list", value=value)
                )
            else:
                raise ValueError(f"Unsupported metadata type: {type(value)}")
    return metadata_entry_list


def _nml_tree_to_wk_tree(
    new_tree: "Tree",
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
        current_node = new_tree.add_node(
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
        new_tree.add_edge(edge.source, edge.target)


def _random_color_rgba() -> tuple[float, float, float, float]:
    """
    A utility to generate a new random RGBA color.
    """
    # https://stackoverflow.com/a/43437435/783758

    h, s, l = (  # noqa: E741 Ambiguous variable name
        np.random.random(),
        0.5 + np.random.random() / 2.0,
        0.4 + np.random.random() / 5.0,
    )
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return (r, g, b, 1.0)


def annotation_to_nml(
    annotation: "Annotation",
) -> wknml.Nml:
    nml_parameters = wknml.Parameters(
        dataset_id=annotation.dataset_id,
        name=annotation.dataset_name,
        scale=annotation.voxel_size,
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
        for tree in annotation.skeleton.flattened_trees()
        for node in tree.nodes
        if node.comment is not None
    ]

    branchpoints = [
        wknml.Branchpoint(node.id, node.time)
        for tree in annotation.skeleton.flattened_trees()
        for node in tree.nodes
        if node.is_branchpoint
    ]

    trees = []

    for tree in sorted(annotation.skeleton.flattened_trees(), key=lambda g: g.id):
        nodes, edges = _extract_nodes_and_edges_from_tree(tree)
        color = tree.color or _random_color_rgba()
        name = tree.name or f"tree{tree.id}"
        metadata = _dict_to_metadata_entry_list(tree.metadata) or []

        trees.append(
            wknml.Tree(
                nodes=nodes,
                edges=edges,
                id=tree.id,
                name=name,
                groupId=tree.group.id if tree.group != annotation.skeleton else None,
                color=color,
                metadata=metadata,
            )
        )

    groups = annotation.skeleton.as_nml_group().children

    volumes = [
        wknml.Volume(
            id=volume.id,
            location=volume._default_zip_name(),
            fallback_layer=volume.fallback_layer_name,
            name=volume.name,
            segments=[
                wknml.Segment(
                    id=segment_id,
                    name=segment_info.name,
                    anchor_position=segment_info.anchor_position,
                    color=segment_info.color,
                    metadata=_dict_to_metadata_entry_list(segment_info.metadata),
                )
                for segment_id, segment_info in volume.segments.items()
            ],
            format=str(volume.data_format),
            largest_segment_id=volume.largest_segment_id,
        )
        for volume in annotation._volume_layers
    ]

    meta = [
        wknml.Meta(name=key, content=value)
        for key, value in annotation.metadata.items()
    ]
    if annotation.owner_name is not None:
        meta.append(wknml.Meta(name="username", content=annotation.owner_name))
    if annotation.annotation_id is not None:
        meta.append(wknml.Meta(name="annotationId", content=annotation.annotation_id))

    nml = wknml.Nml(
        meta=meta,
        parameters=nml_parameters,
        trees=trees,
        branchpoints=branchpoints,
        comments=comments,
        groups=groups,
        volumes=volumes,
    )

    return nml


def _extract_nodes_and_edges_from_tree(
    graph: nx.Graph,
) -> tuple[list[wknml.Node], list[wknml.Edge]]:
    """
    A utility to convert a single [NetworkX graph object](https://networkx.org/) into a list of `NmlNode` objects and `Edge` objects.

    Return
        tuple[list[NmlNode], list[Edge]]: A tuple containing both all nodes and all edges
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
