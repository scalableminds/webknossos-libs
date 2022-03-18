import argparse
import logging
import os
import re
import textwrap
from collections import namedtuple
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator, List, Set, Tuple

import numpy as np

import webknossos as wk
from webknossos.dataset import Layer, MagView
from webknossos.geometry import BoundingBox, Mag, Vec3Int
from webknossos.utils import add_verbose_flag, setup_logging, time_start, time_stop

logger = logging.getLogger(__name__)


NEIGHBORS = [
    Vec3Int(1, 0, 0),
    Vec3Int(-1, 0, 0),
    Vec3Int(0, 1, 0),
    Vec3Int(0, -1, 0),
    Vec3Int(0, 0, 1),
    Vec3Int(0, 0, -1),
]

FloodFillBbox = namedtuple(
    "FloodFillBbox",
    ["bounding_box", "seed_position", "source_id", "target_id", "timestamp"],
)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(
            """\
         Example usage:
         The following invocation will create a new dataset at "some/path/new_dataset"
         which will be a shallow copy of "existing/dataset" with the exception that
         the "segmentation" layer will have the volume data from "annotation/data"
         merged in. Additionally, the partial flood-fills which are denoted in
         "explorational.nml" will be continued/globalized.

         python -m script_collection.globalize_floodfill \\
            --output_path some/path/new_dataset \\
            --segmentation_layer_path existing/dataset/segmentation \\
            --volume_path annotation/data \\
            --nml_path explorational.nml
         """
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--volume_path",
        "-v",
        help="Directory containing the volume tracing.",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--segmentation_layer_path",
        "-s",
        help="Directory containing the segmentation layer.",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--nml_path",
        "-n",
        help="NML that contains the bounding boxes",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output_path", "-o", help="Output directory", type=Path, required=True
    )

    add_verbose_flag(parser)

    return parser


def get_chunk_pos_and_offset(
    global_position: Vec3Int, chunk_shape: Vec3Int
) -> Tuple[Vec3Int, Vec3Int]:
    offset = global_position % chunk_shape
    return (
        global_position - offset,
        offset,
    )


def execute_floodfill(
    data_mag: MagView,
    seed_position: Vec3Int,
    already_processed_bbox: BoundingBox,
    source_id: int,
    target_id: int,
) -> None:
    cube_size = data_mag.info.shard_size
    cube_bbox = BoundingBox(Vec3Int(0, 0, 0), cube_size)
    chunk_with_relative_seed: List[Tuple[Vec3Int, Vec3Int]] = [
        get_chunk_pos_and_offset(seed_position, cube_size)
    ]

    # The `is_visited` variable is used to know which parts of the already processed bbox
    # were already traversed. Outside of that bounding box, the actual data already
    # is an indicator of whether the flood-fill has reached a voxel.
    is_visited = np.zeros(already_processed_bbox.size.to_tuple(), dtype=np.uint8)
    chunk_count = 0

    while len(chunk_with_relative_seed) > 0:
        chunk_count += 1
        if chunk_count % 10000 == 0:
            logger.info(f"Handled seed positions {chunk_count}")

        dirty_bucket = False
        current_cube, relative_seed = chunk_with_relative_seed.pop()
        global_seed = current_cube + relative_seed

        # Only reading one voxel for the seed can be up to 30,000 times faster
        # which is very relevent, since the chunk doesn't need to be traversed
        # if the seed voxel was already covered.
        value_at_seed_position = data_mag.read(current_cube + relative_seed, (1, 1, 1))

        if value_at_seed_position == source_id or (
            already_processed_bbox.contains(global_seed)
            and value_at_seed_position == target_id
            and not is_visited[global_seed - already_processed_bbox.topleft]
        ):
            logger.info(
                f"Handling chunk {chunk_count} with current cube {current_cube}"
            )
            time_start("read data")
            cube_data = data_mag.read(current_cube, cube_size)
            cube_data = cube_data[0, :, :, :]
            time_stop("read data")

            seeds_in_current_chunk: Set[Vec3Int] = set()
            seeds_in_current_chunk.add(relative_seed)

            time_start("traverse cube")
            while len(seeds_in_current_chunk) > 0:
                current_relative_seed = seeds_in_current_chunk.pop()
                current_global_seed = current_cube + current_relative_seed
                if already_processed_bbox.contains(current_global_seed):
                    is_visited[current_global_seed - already_processed_bbox.topleft] = 1

                if cube_data[current_relative_seed] != target_id:
                    cube_data[current_relative_seed] = target_id
                    dirty_bucket = True

                # check neighbors
                for neighbor in NEIGHBORS:
                    neighbor_pos = current_relative_seed + neighbor

                    global_neighbor_pos = current_cube + neighbor_pos
                    if already_processed_bbox.contains(global_neighbor_pos):
                        if is_visited[
                            global_neighbor_pos - already_processed_bbox.topleft
                        ]:
                            continue
                    if cube_bbox.contains(neighbor_pos):
                        if cube_data[neighbor_pos] == source_id or (
                            already_processed_bbox.contains(global_neighbor_pos)
                            and cube_data[neighbor_pos] == target_id
                        ):
                            seeds_in_current_chunk.add(neighbor_pos)
                    else:
                        chunk_with_relative_seed.append(
                            get_chunk_pos_and_offset(global_neighbor_pos, cube_size)
                        )
            time_stop("traverse cube")

            if dirty_bucket:
                time_start("write chunk")
                data_mag.write(cube_data, current_cube)
                time_stop("write chunk")


@contextmanager
def temporary_annotation_view(volume_annotation_path: Path) -> Iterator[Layer]:

    """
    Given a volume annotation path, create a temporary dataset which
    contains the volume annotation via a symlink. Yield the layer
    so that one can work with the annotation as a wk.Dataset.
    """

    with TemporaryDirectory() as tmp_annotation_dir:
        tmp_annotation_dataset_path = (
            Path(tmp_annotation_dir) / "tmp_annotation_dataset"
        )

        input_annotation_dataset = wk.Dataset(
            str(tmp_annotation_dataset_path), scale=(1, 1, 1), exist_ok=True
        )

        # Ideally, the following code would be used, but there are two problems:
        # - save_volume_annotation cannot deal with the
        #   new named volume annotation layers, yet
        # - save_volume_annotation tries to read the entire data (beginning from 0, 0, 0)
        #   to infer the largest_segment_id which can easily exceed the available RAM.
        #
        # volume_annotation = open_annotation(volume_annotation_path)
        # input_annotation_layer = volume_annotation.save_volume_annotation(
        #     input_annotation_dataset, "volume_annotation"
        # )

        os.symlink(volume_annotation_path, tmp_annotation_dataset_path / "segmentation")
        input_annotation_layer = input_annotation_dataset.add_layer_for_existing_files(
            layer_name="segmentation",
            category="segmentation",
            largest_segment_id=0,  # This is incorrect, but for globalize_floodfill not relevant.
        )

        yield input_annotation_layer


def merge_with_fallback_layer(
    output_path: Path,
    volume_annotation_path: Path,
    segmentation_layer_path: Path,
) -> MagView:

    assert not output_path.exists(), f"Dataset at {output_path} already exists"

    # Prepare output dataset by creatign a shallow copy of the dataset
    # determined by segmentation_layer_path, but do a deep copy of
    # segmentation_layer_path itself (so that we can mutate it).
    input_segmentation_dataset = wk.Dataset.open(segmentation_layer_path.parent)
    time_start("Prepare output dataset")
    output_dataset = input_segmentation_dataset.shallow_copy_dataset(
        output_path,
        name=output_path.name,
        make_relative=True,
        layers_to_ignore=[segmentation_layer_path.name],
    )
    output_layer = output_dataset.add_copy_layer(
        segmentation_layer_path, segmentation_layer_path.name
    )
    time_stop("Prepare output dataset")

    input_segmentation_mag = input_segmentation_dataset.get_layer(
        segmentation_layer_path.name
    ).get_best_mag()
    with temporary_annotation_view(volume_annotation_path) as input_annotation_layer:
        input_annotation_mag = input_annotation_layer.get_best_mag()
        bboxes = [
            bbox.in_mag(input_annotation_mag._mag)
            for bbox in input_annotation_mag.get_bounding_boxes_on_disk()
        ]
        output_mag = output_layer.get_mag(input_segmentation_mag.mag)

        cube_size = output_mag.info.chunk_size[0] * output_mag.info.chunks_per_shard[0]
        chunks_with_bboxes = BoundingBox.group_boxes_with_aligned_mag(
            bboxes, Mag(cube_size)
        )

        assert (
            input_annotation_mag.info.chunks_per_shard == Vec3Int.ones()
        ), "volume annotation must have file_len=1"
        assert (
            input_annotation_mag.info.voxel_type
            == input_segmentation_mag.info.voxel_type
        ), "Volume annotation must have same dtype as fallback layer"

        chunk_count = 0
        for chunk, bboxes in chunks_with_bboxes.items():
            chunk_count += 1
            logger.info(f"Processing chunk {chunk_count}...")

            time_start("Read chunk")
            data_buffer = output_mag.read(chunk.topleft, chunk.size)[0, :, :, :]
            time_stop("Read chunk")

            time_start("Read/merge bboxes")
            for bbox in bboxes:
                read_data = input_annotation_mag.read(bbox.topleft, bbox.size)
                data_buffer[bbox.offset(-chunk.topleft).to_slices()] = read_data
            time_stop("Read/merge bboxes")

            time_start("Write chunk")
            output_mag.write(data_buffer, chunk.topleft)
            time_stop("Write chunk")
    return output_mag


def main(args: argparse.Namespace) -> None:

    # Use the skeleton API to read the bounding boxes once
    # https://github.com/scalableminds/webknossos-libs/issues/482 is done.
    nml_regex = re.compile(
        r'<userBoundingBox .*name="Limits of flood-fill \(source_id=(\d+), target_id=(\d+), seed=([\d,]+), timestamp=(\d+)\)".*topLeftX="(\d+)" topLeftY="(\d+)" topLeftZ="(\d+)" width="(\d+)" height="(\d+)" depth="(\d+)" />'
    )

    bboxes: List[FloodFillBbox] = []
    nml_file = open(args.nml_path, "r", encoding="utf-8")
    lines = nml_file.readlines()
    nml_file.close()
    for line in lines:
        matches = nml_regex.findall(line)
        for match in matches:
            # each match is a tuple of (source_id, target_id, seed, timestamp, top_left_x, top_left_y, top_left_z, width, height, depth
            bboxes.append(
                FloodFillBbox(
                    bounding_box=BoundingBox(
                        (match[4], match[5], match[6]), (match[7], match[8], match[9])
                    ),
                    seed_position=Vec3Int(match[2].split(",")),
                    source_id=int(match[0]),
                    target_id=int(match[1]),
                    timestamp=int(match[3]),
                )
            )
    bboxes = sorted(bboxes, key=lambda x: x.timestamp)

    time_start("Merge with fallback layer")
    data_mag = merge_with_fallback_layer(
        args.output_path,
        args.volume_path,
        args.segmentation_layer_path,
    )
    time_stop("Merge with fallback layer")

    time_start("All floodfills")
    for floodfill in bboxes:
        time_start("Floodfill")
        execute_floodfill(
            data_mag,
            floodfill.seed_position,
            floodfill.bounding_box,
            floodfill.source_id,
            floodfill.target_id,
        )
        time_stop("Floodfill")
    time_stop("All floodfills")

    time_start("Recompute downsampled mags")
    data_mag.layer.redownsample()
    time_stop("Recompute downsampled mags")


if __name__ == "__main__":
    parsed_args = create_parser().parse_args()
    setup_logging(parsed_args)

    main(parsed_args)
