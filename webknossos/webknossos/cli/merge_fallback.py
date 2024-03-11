"""This module merges a volume annotation layer with its fallback layer."""

import logging
from argparse import Namespace
from multiprocessing import cpu_count
from typing import Any, List, Optional, Tuple

import typer
from cluster_tools import Executor
from rich.progress import track
from typing_extensions import Annotated

from ..annotation import Annotation
from ..dataset import Dataset, MagView
from ..geometry import BoundingBox, Mag
from ..utils import get_executor_for_args
from ._utils import DistributionStrategy, parse_path


def main(
    *,
    target: Annotated[
        Any,
        typer.Argument(
            help="Path to your WEBKNOSSOS output dataset.",
            show_default=False,
            parser=parse_path,
        ),
    ],
    source_annotation: Annotated[
        Any,
        typer.Argument(
            help="Path to your WEBKNOSSOS zip annotation",
            show_default=False,
            parser=parse_path,
        ),
    ],
    dataset_directory: Annotated[
        Any,
        typer.Argument(
            help="Path to your WEBKNOSSOS dataset folder.",
            show_default=False,
            parser=parse_path,
        ),
    ],
    volume_layer_name: Annotated[
        Optional[str],
        typer.Option(help="Name of the volume layer to merge with fallback layer."),
    ] = None,
    jobs: Annotated[
        int,
        typer.Option(
            help="Number of processes to be spawned.",
            rich_help_panel="Executor options",
        ),
    ] = cpu_count(),
    distribution_strategy: Annotated[
        DistributionStrategy,
        typer.Option(
            help="Strategy to distribute the task across CPUs or nodes.",
            rich_help_panel="Executor options",
        ),
    ] = DistributionStrategy.MULTIPROCESSING,
    job_resources: Annotated[
        Optional[str],
        typer.Option(
            help="Necessary when using slurm as distribution strategy. Should be a JSON string "
            '(e.g., --job_resources=\'{"mem": "10M"}\')\'',
            rich_help_panel="Executor options",
        ),
    ] = None,
) -> None:
    """Merges a given WEBKNOSSOS annotation."""

    executor_args = Namespace(
        jobs=jobs,
        distribution_strategy=distribution_strategy.value,
        job_resources=job_resources,
    )

    annotation = Annotation.load(source_annotation)
    annotation_volumes = list(annotation.get_volume_layer_names())

    output_dataset = Dataset(
        target,
        voxel_size=annotation.voxel_size,
    )
    assert len(annotation_volumes) > 0, "Annotation does not contain any volume layers!"

    if volume_layer_name is not None:
        assert (
            volume_layer_name in annotation_volumes
        ), f'Volume layer name "{volume_layer_name}" not found in annotation'
    else:
        assert (
            len(annotation_volumes) == 1
        ), "Volume layer name was not provided and more than one volume layer found in annotation"
        volume_layer_name = annotation_volumes[0]

    volume_layer = annotation._get_volume_layer(volume_layer_name=volume_layer_name)
    fallback_layer_name = volume_layer.fallback_layer_name

    if fallback_layer_name is None:
        logging.info("No fallback layer found, save annotation as dataset.")
        annotation.export_volume_layer_to_dataset(output_dataset)

    else:
        fallback_dataset_path = dataset_directory / annotation.dataset_name
        fallback_layer = Dataset.open(fallback_dataset_path).get_layer(
            fallback_layer_name
        )

        if volume_layer.zip is None:
            logging.info("No volume annotation found. Copy fallback layer.")
            with get_executor_for_args(args=executor_args) as executor:
                output_dataset.add_copy_layer(
                    fallback_layer, compress=True, executor=executor
                )

        else:
            tmp_annotation_layer_name = f"{annotation.name}-TMP"
            logging.info(
                f"Unpack annotation layer {volume_layer_name} temporarily in {output_dataset.name} as {tmp_annotation_layer_name}"
            )
            # NOTE(erjel): Cannot use "temporary_volume_layer_copy" here, since tmp folders
            # might not be accessible from slurm compute nodes.
            input_annotation_layer = annotation.export_volume_layer_to_dataset(
                output_dataset,
                layer_name=tmp_annotation_layer_name,
                volume_layer_name=volume_layer_name,
            )

            input_annotation_mag = input_annotation_layer.get_finest_mag()
            fallback_mag = fallback_layer.get_mag(input_annotation_mag.mag)

            logging.info(f"Create layer {fallback_layer.name} in {output_dataset.path}")
            output_layer = output_dataset.add_layer_like(
                fallback_layer, fallback_layer.name
            )

            with get_executor_for_args(args=executor_args) as executor:
                logging.info(
                    f"Copy Mag {fallback_mag.mag} from {fallback_layer.path} to {output_layer.path}"
                )
                output_mag = output_layer.add_copy_mag(
                    fallback_mag,
                    compress=True,
                    executor=executor,
                )

                merge_mags(output_mag, input_annotation_mag, executor)

            logging.info("Delete temporary annotation layer")
            output_dataset.delete_layer(tmp_annotation_layer_name)
            logging.info("Done.")


def merge_mags(
    output_mag: MagView,
    input_annotation_mag: MagView,
    executor: Executor,
) -> None:
    assert all(
        input_annotation_mag.info.chunks_per_shard.to_np() == 1
    ), "volume annotation must have file_len=1"
    assert (
        input_annotation_mag.info.voxel_type == output_mag.info.voxel_type
    ), "Volume annotation must have same dtype as fallback layer"
    assert (
        input_annotation_mag.mag == output_mag.mag
    ), f"Volume annotation mag {input_annotation_mag.mag} must match the fallback layer mag {output_mag.mag}"

    logging.info("Scan disk for annotation shards.")
    bboxes = list(bbox for bbox in input_annotation_mag.get_bounding_boxes_on_disk())

    logging.info(f"Grouping {len(bboxes)} bboxes according to output shards.")
    shards_with_bboxes = BoundingBox.group_boxes_with_aligned_mag(
        bboxes, Mag(output_mag.info.shard_shape * output_mag.mag)
    )

    args = [
        (input_annotation_mag, output_mag, shard, bboxes)
        for shard, bboxes in shards_with_bboxes.items()
    ]

    logging.info(f"Merging {len(args)} shards.")
    executor.map(merge_chunk, args)


def merge_chunk(args: Tuple[MagView, MagView, BoundingBox, List[BoundingBox]]) -> None:
    mag_in, mag_out, shard, bboxes = args
    data_buffer = mag_out.read(absolute_bounding_box=shard)[0]

    for bbox in track(bboxes, description="Processing..."):
        read_data = mag_in.read(absolute_bounding_box=bbox)[0]
        data_buffer[bbox.offset(-shard.topleft).in_mag(mag_in.mag).to_slices()] = (
            read_data
        )

    mag_out.write(data_buffer, absolute_offset=shard.topleft)
