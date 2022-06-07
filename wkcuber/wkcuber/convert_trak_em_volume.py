import logging
import cairosvg
from argparse import ArgumentParser, Namespace
from pathlib import Path
import xml.etree.ElementTree as ET
from collections import defaultdict
import os

from svgpathtools import parse_path, wsvg

from ._internal.utils import (
    add_verbose_flag,
    setup_logging,
    setup_warnings,
    get_executor_for_args,
    add_distribution_flags,
    wait_and_ensure_success,
)
from webknossos.skeleton import Skeleton


def parse_vec2_int(value: str) -> [int, int]:
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) == 2:
        return parts
    else:
        raise TypeError(f"Cannot convert `{value}` to a list of two ints.")


def create_parser() -> ArgumentParser:
    parser = ArgumentParser()

    parser.add_argument(
        "source_path", help="TrakEM file containing the volume annotations", type=Path
    )

    parser.add_argument(
        "target_path",
        help="Output directory for the generated dataset",
        type=Path,
    )

    parser.add_argument(
        "--no-segments",
        help="Do not convert included segments",
        dest="no_segments",
        action="store_true",
    )

    parser.add_argument(
        "--no-skeletons",
        help="Do not conver included skeletons",
        dest="no_skeletons",
        action="store_true",
    )

    parser.add_argument(
        "--offset",
        "-o",
        default=[0, 0],
        type=parse_vec2_int,
        help="Offset that should be added to all skeletons.",
    )

    add_verbose_flag(parser)
    add_distribution_flags(parser)

    return parser


def get_rgb_from_uint24(uint24):
    blue = uint24 & 255
    green = (uint24 >> 8) & 255
    red = (uint24 >> 16) & 255
    return [red, green, blue]


def convert_slice(args):

    z_slice, paths, max_width, max_height, target_path = args

    if paths is not None:
        svg_paths, segments, offsets = zip(
            *[
                (parse_path(svg_path), segment, offset)
                for segment, offset, svg_path in paths
            ]
        )
        attributes = [
            {
                "fill": f"rgb({segment % 256},0,0)",
                "shape-rendering": "crispEdges",
                "transform": f"translate({offset[0]} {offset[1]})",
            }
            for segment, offset in zip(segments, offsets)
        ]

        svg_file_path = str(target_path / f"temp/{z_slice:05}.svg")
        wsvg(
            svg_paths,
            attributes=attributes,
            # svg_attributes=svg_attributes,
            filename=svg_file_path,
            dimensions=(max_width, max_height),
        )

        png_file_path = str(target_path / f"temp/png_dataset/{z_slice:05}.png")
        cairosvg.svg2png(url=svg_file_path, write_to=png_file_path)

    else:
        # Link an empty png file, because the wkcuber converter doesn't support missing slices
        os.symlink(
            target_path.resolve() / "empty.png",
            target_path / f"temp/png_dataset/{z_slice:05}.png",
        )


def main(args: Namespace) -> None:

    if not args.no_segments:

        with open(args.source_path) as file:
            path_dict, max_width, max_height = parse_segments_from_trakem_xml(
                file, args.offset
            )

        max_z_slice = max(path_dict.keys())

        print(
            f"Found {len(path_dict)} slices with a volume annotation with maximum width {max_width} and height {max_height}."
        )

        os.makedirs(args.target_path / "temp", exist_ok=True)
        os.makedirs(args.target_path / "temp/png_dataset", exist_ok=True)

        with get_executor_for_args(args) as executor:

            job_args = [
                (
                    z_slice,
                    path_dict.get(z_slice, None),
                    max_width,
                    max_height,
                    args.target_path,
                )
                for z_slice in range(max_z_slice + 1)
            ]

            wait_and_ensure_success(
                executor.map_to_futures(
                    convert_slice,
                    job_args,
                ),
                progress_desc=f"Converting {len(path_dict)} slices",
            )

    if not args.no_skeletons:

        with open(args.source_path) as file:
            annotation = parse_skeletons_from_trakem_xml(file, args.offset)

        print(
            f"Found {len(list(annotation.flattened_trees()))} skeletons with {annotation.get_total_node_count()} nodes in {len(list(annotation.flattened_groups()))} groups."
        )

        source_file_name = args.source_path.stem
        annotation.save(args.target_path / f"{source_file_name}.nml")


def parse_skeletons_from_trakem_xml(file, global_offset):
    current_offset = (0, 0)
    current_node_stack = []
    current_tree = None
    oid_to_slice = {}
    oid_to_tree_id = {}
    oid_to_current_group = {}
    scale = (1, 1, 1)

    skeleton = Skeleton(scale, "unknown dataset")
    current_group_stack = [skeleton]

    # First pass, parse OIDs
    for event, elem in ET.iterparse(file, events=("start", "end")):
        if event == "start":
            if elem.tag == "t2_calibration":
                x = float(elem.get("pixelWidth"))
                y = float(elem.get("pixelHeight"))
                z = float(elem.get("pixelDepth"))
                scale = (x, y, z)
            elif elem.tag == "t2_layer":
                oid = int(elem.get("oid"))
                thickness = float(elem.get("thickness"))
                z = float(elem.get("z"))
                z_vx = int(z / thickness)
                assert z_vx == (
                    z / thickness
                ), f"z / thickness ({z} / {thickness}) is not an integer"
                # Subtract 1, because webKnossos is 0-indexed and TrakEM is 1-indexed
                oid_to_slice[oid] = z_vx - 1
            elif elem.tag == "anything" or elem.tag == "cell_type":
                group_name = elem.get("title")
                current_group = current_group_stack[-1].add_group(group_name)
                current_group_stack.append(current_group)
            elif elem.tag == "treeline":
                oid = int(elem.get("oid"))
                tree_id = int(elem.get("id"))
                oid_to_tree_id[oid] = tree_id
                oid_to_current_group[oid] = current_group_stack[-1]
        elif event == "end":
            if elem.tag == "anything" or elem.tag == "cell_type":
                current_group_stack.pop()
                # The group stack should always contain the skeleton
                assert len(current_group_stack) > 0, "Group stack is empty"

    skeleton.voxel_size = scale

    # Second pass, parse skeleton
    file.seek(0)
    for event, elem in ET.iterparse(file, events=("start", "end")):
        if event == "start":
            if elem.tag == "t2_treeline":
                oid = int(elem.get("oid"))
                name = elem.get("title")
                tree_id = oid_to_tree_id[oid]
                current_group = oid_to_current_group[oid]

                # transform(x11, x12, x13, x21, x22, x23)
                transform_str = elem.get("transform")
                transform_tuple = list(
                    map(int, map(float, transform_str[7:-1].split(",")))
                )
                assert (
                    len(transform_tuple) == 6
                ), f"Transform has {len(transform_tuple)} instead of 6 entries"
                current_offset = (
                    transform_tuple[-2] + global_offset[0],
                    transform_tuple[-1] + global_offset[1],
                )

                current_tree = current_group.add_tree(name)
            elif elem.tag == "t2_node":
                if current_tree is None:
                    continue

                x = int(float(elem.get("x")))
                y = int(float(elem.get("y")))
                oid = int(elem.get("lid"))
                z = oid_to_slice[oid]

                r = elem.get("r")

                node_position = (current_offset[0] + x, current_offset[1] + y, z)

                node = current_tree.add_node(node_position)

                if r is not None:
                    node.radius = float(r)

                if len(current_node_stack) > 0:
                    current_tree.add_edge(current_node_stack[-1], node)

                current_node_stack.append(node)
            elif elem.tag == "t2_tag":
                if current_tree is None:
                    continue

                comment = elem.get("name")
                key = elem.get("key")
                current_node_stack[-1].comment = comment

                # The T key (TODO) seems to indicate a branchpoint
                if key == "T":
                    current_node_stack[-1].is_branchpoint = True
        elif event == "end":
            if elem.tag == "t2_treeline":
                current_offset = None
                assert len(current_node_stack) == 0, "Node stack is not empty"
                current_group = None
                current_tree = None
            elif elem.tag == "t2_node":
                if current_tree is None:
                    continue

                current_node_stack.pop()

    return skeleton


def parse_segments_from_trakem_xml(file, global_offset):
    max_width = 0
    max_height = 0
    current_segment = None
    current_offset = (0, 0)
    current_slice = None
    oid_to_path = defaultdict(list)
    oid_to_slice = {}
    oid_to_segment = {}
    segments = set()

    for event, elem in ET.iterparse(file, events=("start", "end")):
        if event == "start":
            if elem.tag == "t2_areatree":
                oid = int(elem.get("oid"))
                current_segment = oid_to_segment[oid]
                segments.add(current_segment)
                width = int(float(elem.get("width")))
                height = int(float(elem.get("height")))
                # transform(x11, x12, x13, x21, x22, x23)
                transform_str = elem.get("transform")
                transform_tuple = list(
                    map(int, map(float, transform_str[7:-1].split(",")))
                )
                assert (
                    len(transform_tuple) == 6
                ), f"Transform has {len(transform_tuple)} instead of 6 entries"
                current_offset = (
                    transform_tuple[-2] + global_offset[0],
                    transform_tuple[-1] + global_offset[1],
                )

                max_width = max(max_width, width + current_offset[0])
                max_height = max(max_height, height + current_offset[1])

            elif elem.tag == "t2_node":
                current_slice = int(elem.get("lid"))
            elif elem.tag == "t2_path":
                if current_segment is None:
                    continue

                assert (
                    current_segment is not None
                ), "<t2_path ...> tag needs to be child of a <t2_areatree ...> tag."
                assert (
                    current_slice is not None
                ), "<t2_path ...> tag needs to be child of a <t2_node ...> tag."
                path = elem.get("d")
                oid_to_path[current_slice].append(
                    (current_segment, current_offset, path)
                )
            elif elem.tag == "t2_layer":
                oid = int(elem.get("oid"))
                thickness = float(elem.get("thickness"))
                z = float(elem.get("z"))
                z_vx = int(z / thickness)
                assert z_vx == (
                    z / thickness
                ), f"z / thickness ({z} / {thickness}) is not an integer"
                # Subtract 1, because webKnossos is 0-indexed and TrakEM is 1-indexed
                oid_to_slice[oid] = z_vx - 1
            elif elem.tag == "areatree":
                oid = int(elem.get("oid"))
                segment_id = int(elem.get("id"))
                oid_to_segment[oid] = segment_id
        elif event == "end":
            if elem.tag == "t2_areatree":
                current_segment = None
                current_offset = None
            elif elem.tag == "t2_node":
                current_slice = None

    # The segments ids will be cropped to uint8, because the cuber cannot combine multiple PNG channels
    # into one layer, but instead will write a layer per channel. So the segment ids will be cropped to
    # uint8 and saved in the PNG red channel. Make sure segments are not involuntarily merged by this.
    cropped_segments = [segment_id % 256 for segment_id in segments]
    assert len(cropped_segments) == len(
        set(cropped_segments)
    ), "There are segment ids which do not differ in the lowest 8 bit."

    slice_to_path = {oid_to_slice[oid]: path for oid, path in oid_to_path.items()}

    return slice_to_path, max_width, max_height


if __name__ == "__main__":
    setup_warnings()
    args = create_parser().parse_args()
    setup_logging(args)
    main(args)
