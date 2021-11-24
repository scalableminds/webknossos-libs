from collections import defaultdict
import webknossos as wk
import sys
import os
import re
import time
from argparse import ArgumentParser, Namespace
from collections import namedtuple
from glob import iglob
from os import path
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple
import shutil
from icecream import ic
from webknossos.utils import time_start, time_stop

import numpy as np
import wkw

from webknossos.geometry import BoundingBox, Mag, Vec3Int

BUCKET_SIZE = 32
BUCKET_SHAPE = (BUCKET_SIZE, BUCKET_SIZE, BUCKET_SIZE)
# CUBE_SHAPE = BUCKET_SHAPE
CUBE_SHAPE = (1024, 1024, 1024)
NEIGHBORS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

FloodFillBbox = namedtuple(
    "FloodFillBbox",
    ["bounding_box", "seed_position", "source_id", "target_id", "timestamp"],
)


def create_parser() -> ArgumentParser:
    parser = ArgumentParser()

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

    parser.add_argument("--skip_merge", default=False, action="store_true")

    return parser


def get_bucket_pos_and_offset(
    global_position: Tuple[int, int, int]
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    offset = (
        global_position[0] % CUBE_SHAPE[0],
        global_position[1] % CUBE_SHAPE[1],
        global_position[2] % CUBE_SHAPE[2],
    )
    return (
        substract_positions(global_position, offset),
        offset,
    )


def add_positions(
    a: Tuple[int, int, int], b: Tuple[int, int, int]
) -> Tuple[int, int, int]:
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


def substract_positions(
    a: Tuple[int, int, int], b: Tuple[int, int, int]
) -> Tuple[int, int, int]:
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def inside_bbox(
    point: Tuple[int, int, int],
    bbox_top_left: Tuple[int, int, int] = (0, 0, 0),
    bbox_bottom_right: Tuple[int, int, int] = CUBE_SHAPE,
) -> bool:
    return (
        bbox_top_left[0] <= point[0] < bbox_bottom_right[0]
        and bbox_top_left[1] <= point[1] < bbox_bottom_right[1]
        and bbox_top_left[2] <= point[2] < bbox_bottom_right[2]
    )


def execute_floodfill(
    data_path: Path,
    seed_position: Tuple[int, int, int],
    already_processed_bbox: BoundingBox,
    source_id: int,
    target_id: int,
) -> None:
    # bucketData[neighbourVoxelIndex] == sourceCellId || (isInsideBBox & & bucketData[neighbourVoxelIndex] == targetCellId)
    bucket_and_seed_pos: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = [
        get_bucket_pos_and_offset(seed_position)
    ]
    # bucket_to_seed_pos: Dict[
    #    Tuple[int, int, int], Set[Tuple[int, int, int]]
    # ] = defaultdict(set)
    already_processed_bbox_top_left = already_processed_bbox.topleft.to_tuple()
    already_processed_bbox_bottom_right = already_processed_bbox.bottomright.to_tuple()

    # The `visited` variable is used to know which parts of the already processed bbox
    # were already traversed. Outside of that bounding box, the actual data already
    # is an indicator of whether the flood-fill has reached a voxel.
    visited = np.zeros(
        already_processed_bbox.size.to_tuple(), dtype=np.uint8
    )  # bitarray needs less memory, but new dependency
    bucket_count = 0
    with wkw.Dataset.open(str(data_path / "1")) as wkw_data:
        while len(bucket_and_seed_pos) != 0:
            bucket_count += 1
            if bucket_count % 10000 == 0:
                print("Handled seed positions ", bucket_count)

            dirty_bucket = False
            current_cube, seed_position = bucket_and_seed_pos.pop()
            global_seed_position = add_positions(current_cube, seed_position)

            # Only reading one voxel for the seed is approximately 30.000 times faster
            value_at_seed_position = wkw_data.read(
                Vec3Int(current_cube) + Vec3Int(seed_position), (1, 1, 1)
            )

            if value_at_seed_position == source_id or (
                inside_bbox(
                    global_seed_position,
                    already_processed_bbox_top_left,
                    already_processed_bbox_bottom_right,
                )
                and value_at_seed_position == target_id
                and not visited[
                    substract_positions(
                        global_seed_position, already_processed_bbox_top_left
                    )
                ]
            ):
                print("Handling chunk", bucket_count, "with current cube", current_cube)
                time_start("read data")
                cube_data = wkw_data.read(current_cube, CUBE_SHAPE)
                cube_data = cube_data[0, :, :, :]
                time_stop("read data")

                seeds_in_curr_bucket: Set[Tuple[int, int, int]] = set()
                seeds_in_curr_bucket.add(seed_position)
                time_start("traverse cube")
                while len(seeds_in_curr_bucket) > 0:
                    if len(seeds_in_curr_bucket) % 10000 == 0:
                        print("remaining seeds", len(seeds_in_curr_bucket))
                    seed_pos = seeds_in_curr_bucket.pop()
                    global_seed_pos = add_positions(current_cube, seed_pos)
                    if inside_bbox(
                        global_seed_pos,
                        already_processed_bbox_top_left,
                        already_processed_bbox_bottom_right,
                    ):
                        visited[
                            substract_positions(
                                global_seed_pos,
                                already_processed_bbox_top_left,
                            )
                        ] = 1

                    if cube_data[seed_pos] != target_id:
                        cube_data[seed_pos] = target_id
                        dirty_bucket = True

                    # check neighbors
                    for neighbor in NEIGHBORS:
                        neighbor_pos = add_positions(seed_pos, neighbor)
                        global_neighbor_pos = add_positions(current_cube, neighbor_pos)
                        if inside_bbox(
                            global_neighbor_pos,
                            already_processed_bbox_top_left,
                            already_processed_bbox_bottom_right,
                        ):
                            if visited[
                                substract_positions(
                                    global_neighbor_pos,
                                    already_processed_bbox_top_left,
                                )
                            ]:
                                continue
                        if inside_bbox(neighbor_pos):
                            if cube_data[neighbor_pos] == source_id or (
                                inside_bbox(
                                    global_neighbor_pos,
                                    already_processed_bbox_top_left,
                                    already_processed_bbox_bottom_right,
                                )
                                and cube_data[neighbor_pos] == target_id
                            ):
                                seeds_in_curr_bucket.add(neighbor_pos)
                        else:
                            bucket_and_seed_pos.append(
                                get_bucket_pos_and_offset(global_neighbor_pos)
                            )
                time_stop("traverse cube")
                if dirty_bucket:
                    time_start("write chunk")
                    wkw_data.write(current_cube, cube_data)
                    time_stop("write chunk")


def detect_cube_length(layer_path: Path) -> int:
    if layer_path is not None:
        with wkw.Dataset.open(str(layer_path)) as dataset:
            return dataset.header.block_len * dataset.header.file_len
    raise RuntimeError(
        f"Failed to detect the cube length (for {layer_path}) because the layer_path is None"
    )


# def detect_bbox(layer_path: Path) -> Optional[dict]:
#     # Detect the coarse bounding box of a dataset by iterating
#     # over the WKW cubes
#     def list_files(layer_path: str) -> Iterable[str]:
#         return iglob(path.join(layer_path, "*", "*", "*.wkw"), recursive=True)

#     def parse_cube_file_name(filename: str) -> Tuple[int, int, int]:
#         CUBE_REGEX = re.compile(
#             fr"z(\d+){re.escape(os.path.sep)}y(\d+){re.escape(os.path.sep)}x(\d+)(\.wkw)$"
#         )
#         m = CUBE_REGEX.search(filename)
#         if m is not None:
#             return int(m.group(3)), int(m.group(2)), int(m.group(1))
#         raise RuntimeError(f"Failed to parse cube file name from {filename}")

#     def list_cubes(layer_path: str) -> Iterable[Tuple[int, int, int]]:
#         return (parse_cube_file_name(f) for f in list_files(layer_path))

#     xs, ys, zs = list(zip(*list_cubes(str(layer_path))))

#     min_x, min_y, min_z = min(xs), min(ys), min(zs)
#     max_x, max_y, max_z = max(xs), max(ys), max(zs)

#     cube_length = detect_cube_length(layer_path)
#     if cube_length is None:
#         return None

#     return {
#         "topLeft": [min_x * cube_length, min_y * cube_length, min_z * cube_length],
#         "width": (1 + max_x - min_x) * cube_length,
#         "height": (1 + max_y - min_y) * cube_length,
#         "depth": (1 + max_z - min_z) * cube_length,
#     }


def merge_with_fallback_layer(
    output_path: Path,
    volume_data_path: Path,
    segmentation_layer_path: Path,
) -> None:

    # dataset_bbox = detect_bbox(args.segmentation_layer_path)
    # if dataset_bbox is None:
    #     raise ValueError("Could not detect bbox")

    if output_path.exists():
        print("skipping copying")
    else:
        copy_start = time.time()
        shutil.copytree(segmentation_layer_path, output_path)
        copy_end = time.time()
        print(f"Copying took {copy_end - copy_start}")

    fallback_voxel_type = None
    with wkw.Dataset.open(str(segmentation_layer_path / "1")) as fallback_layer_wkw:
        fallback_voxel_type = fallback_layer_wkw.header.voxel_type

    tmp_annotation_dataset_path = Path("tmp_annotation_dataset")
    input_annotation_dataset = wk.Dataset.get_or_create(
        str(tmp_annotation_dataset_path), scale=(1, 1, 1)
    )
    if not (tmp_annotation_dataset_path / "segmentation").exists():
        os.symlink(volume_data_path, tmp_annotation_dataset_path / "segmentation")
        input_annotation_layer = input_annotation_dataset.add_layer_for_existing_files(
            layer_name="segmentation",
            category="segmentation",
            largest_segment_id=0,  # todo
        )
    else:
        input_annotation_layer = input_annotation_dataset.get_layer("segmentation")
    min_mag = min(input_annotation_layer.mags.keys())
    bboxes = list(input_annotation_layer.get_mag(min_mag).get_bounding_boxes_on_disk())

    chunks_with_bboxes = defaultdict(list)
    for bbox_tuple in bboxes:
        bbox = BoundingBox.from_tuple2(bbox_tuple)
        chunk_key = bbox.align_with_mag(Mag(1024), ceil=True)
        chunks_with_bboxes[chunk_key].append(bbox)

    with wkw.Dataset.open(str(volume_data_path / "1")) as volume_data_wkw:
        assert (
            volume_data_wkw.header.file_len == 1
        ), "volume annotation must have file_len=1"
        assert (
            volume_data_wkw.header.voxel_type == fallback_voxel_type
        ), "Volume annotation must have same dtype as fallback layer"
        with wkw.Dataset.open(
            str(output_path / "1"),
        ) as merged_wkw:
            annotated_count = 0
            # bucket_boxes = list(
            #     BoundingBox.from_wkw_dict(bbox)
            #     .align_with_mag(Mag(BUCKET_SIZE), ceil=True)
            #     .chunk(BUCKET_SHAPE)
            # )
            # total_bucket_count = len(bucket_boxes)
            # todo: only iterate over annotation_bucket_paths
            # annotation_bucket_paths = list_bucket_paths(volume_data_path)
            # for count, bucket_bbox in enumerate(bucket_boxes):

            chunk_count = 0
            for chunk, bboxes in chunks_with_bboxes.items():
                chunk_count += 1
                print(f"Processing chunk {chunk_count}...")

                time_start("Read chunk")
                data_buffer = merged_wkw.read(chunk.topleft.to_np(), chunk.size)[
                    0, :, :, :
                ]
                time_stop("Read chunk")

                # print("data_buffer.shape", data_buffer.shape)

                time_start("Read/merge bboxes")
                for bbox in bboxes:
                    annotated_count += 1
                    read_data = volume_data_wkw.read(bbox.topleft.to_np(), bbox.size)
                    # ic(read_data.shape)
                    # ic(bbox.offset(-chunk.topleft))
                    data_buffer[bbox.offset(-chunk.topleft).to_slices()] = read_data
                time_stop("Read/merge bboxes")

                time_start("Write chunk")
                merged_wkw.write(chunk.topleft.to_np(), data_buffer)
                time_stop("Write chunk")

    # print(
    #     f"Combined {annotated_count} volume-annotated buckets with"
    #     f" {total_bucket_count - annotated_count} from fallback layer."
    # )


# def list_bucket_paths(volume_data_layer_path: Path) -> List[Path]:
#     return [
#         wkw_path.relative_to(volume_data_layer_path)
#         for wkw_path in volume_data_layer_path.rglob("*/*/*.wkw")
#     ]


# def bucket_path_for_pos(position: Vec3Int) -> Path:
#     x_bucket, y_bucket, z_bucket = position // BUCKET_SIZE
#     return Path(f"z{z_bucket}/y{y_bucket}/x{x_bucket}.wkw")


def main(args: Namespace) -> None:

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

    if not args.skip_merge:
        start = time.time()
        merge_with_fallback_layer(
            args.output_path,
            args.volume_path,
            args.segmentation_layer_path,
        )
        print("Combining data took ", time.time() - start)
    # return
    overall_start = time.time()
    for floodfill in bboxes:
        start = time.time()
        execute_floodfill(
            args.output_path,
            floodfill.seed_position.to_tuple(),
            floodfill.bounding_box,
            floodfill.source_id,
            floodfill.target_id,
        )
        print("Current floodfill took ", time.time() - start)
    print("All floodfills took ", time.time() - overall_start)


if __name__ == "__main__":
    parsed_args = create_parser().parse_args()

    main(parsed_args)
