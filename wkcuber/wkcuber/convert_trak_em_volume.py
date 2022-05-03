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

uint24_max = 2 ** 24


def create_parser() -> ArgumentParser:
    parser = ArgumentParser()

    parser.add_argument(
        "source_path", help="TrakEM file containing the volume annotations", type=Path
    )

    parser.add_argument(
        "target_path",
        help="Output directory for the generated dataset.",
        type=Path,
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
                "fill": f"rgb({','.join(map(str, get_rgb_from_uint24(segment)))})",
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

    with open(args.source_path) as file:
        path_dict, max_width, max_height = parse_trakem_xml(file)

    max_z_slice = max(path_dict.keys())

    print(
        f"Found {len(path_dict)} slices with a volume annotation with maximum width {max_width} and height {max_height}"
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


def parse_trakem_xml(file):
    max_width = 0
    max_height = 0
    current_segment = None
    current_offset = (0, 0)
    current_slice = None
    oid_to_path = defaultdict(list)
    oid_to_slice = {}

    for event, elem in ET.iterparse(file, events=("start", "end")):
        if event == "start":
            if elem.tag == "t2_areatree":
                current_segment = int(elem.get("oid"))
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
                current_offset = (transform_tuple[-2], transform_tuple[-1])

                max_width = max(max_width, width + current_offset[0])
                max_height = max(max_height, height + current_offset[1])

                assert (
                    current_segment < uint24_max
                ), f"Segment ID {current_segment} doesn't fit into uint24"
            elif elem.tag == "t2_node":
                current_slice = int(elem.get("lid"))
            elif elem.tag == "t2_path":
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
                oid_to_slice[oid] = z_vx
        elif event == "end":
            if elem.tag == "t2_areatree":
                current_segment = None
                current_offset = None
            elif elem.tag == "t2_node":
                current_slice = None

    slice_to_path = {oid_to_slice[oid]: path for oid, path in oid_to_path.items()}

    return slice_to_path, max_width, max_height


if __name__ == "__main__":
    setup_warnings()
    args = create_parser().parse_args()
    setup_logging(args)
    main(args)
