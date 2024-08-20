"""This module checks equality of two different WEBKNOSSOS datasets."""

import logging
from argparse import Namespace
from multiprocessing import cpu_count
from typing import Any, Optional

import typer
from typing_extensions import Annotated

from ..dataset import Dataset, Layer
from ..utils import get_executor_for_args
from ._utils import DistributionStrategy, parse_path

logger = logging.getLogger(__name__)


def main(
    *,
    source: Annotated[
        Any,
        typer.Argument(
            help="Path to your first WEBKNOSSOS dataset.",
            show_default=False,
            parser=parse_path,
        ),
    ],
    target: Annotated[
        Any,
        typer.Argument(
            help="Path to your second WEBKNOSSOS dataset.",
            show_default=False,
            parser=parse_path,
        ),
    ],
    layer_name: Annotated[
        Optional[str],
        typer.Option(
            help="Name of the layer to compare (if not provided, all layers are compared)."
        ),
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
            '(e.g., --job-resources=\'{"mem": "10M"}\')\'',
            rich_help_panel="Executor options",
        ),
    ] = None,
) -> None:
    """Check equality of two WEBKNOSSOS datasets."""

    executor_args = Namespace(
        jobs=jobs,
        distribution_strategy=distribution_strategy.value,
        job_resources=job_resources,
    )

    source_dataset = Dataset.open(source)
    target_dataset = Dataset.open(target)
    source_layer_names = set(source_dataset.layers.keys())
    target_layer_names = set(target_dataset.layers.keys())

    layer_names = list(source_layer_names)

    try:
        if layer_name is not None:
            assert (
                layer_name in source_layer_names
            ), f"Provided layer {layer_name} does not exist in source dataset."
            assert (
                layer_name in target_layer_names
            ), f"Provided layer {layer_name} does not exist in target dataset."
            layer_names = [layer_name]

        else:
            assert (
                source_layer_names == target_layer_names
            ), f"The provided input datasets have different \
    layers: {source_layer_names} != {target_layer_names}"

        for name in layer_names:
            compare_layers(
                source_dataset.get_layer(name),
                target_dataset.get_layer(name),
                executor_args,
            )

        print(
            f"The datasets {source} and {target} are equal \
    (with regard to the layers: {layer_names})"
        )
    except AssertionError as err:
        print(f"The datasets are not equal: {err}")
        exit(1)


def compare_layers(
    source_layer: Layer,
    target_layer: Layer,
    executor_args: Namespace,
) -> None:
    """Compares one layer with another layer"""

    layer_name = source_layer.name
    logging.info("Checking layer_name: %s", layer_name)

    assert (
        source_layer.bounding_box == target_layer.bounding_box
    ), f"The bounding boxes of '{layer_name}' layer of source and target \
are not equal: {source_layer.bounding_box} != {target_layer.bounding_box}"

    source_mags = set(source_layer.mags.keys())
    target_mags = set(target_layer.mags.keys())

    assert (
        source_mags == target_mags
    ), f"The mags of '{layer_name}' layer of source and target are not equal: \
{source_mags} != {target_mags}"

    for mag in source_mags:
        source_mag = source_layer.mags[mag]
        target_mag = target_layer.mags[mag]

        logging.info("Start verification of %s in mag %s", layer_name, mag)
        with get_executor_for_args(args=executor_args) as executor:
            assert source_mag.content_is_equal(
                target_mag, executor=executor
            ), f"The contents of {source_mag} and {target_mag} differ."
