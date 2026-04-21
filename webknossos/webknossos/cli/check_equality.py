"""This module checks equality of two different WEBKNOSSOS datasets."""

import logging
from typing import Annotated, Any

import typer
from cluster_tools import Executor

from ..dataset import Dataset, Layer
from ._utils import (
    DEFAULT_JOBS,
    DistributionStrategy,
    DistributionStrategyOption,
    JobResourcesOption,
    JobsOption,
    get_executor_for_args,
    parse_path,
)

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
        str | None,
        typer.Option(
            help="Name of the layer to compare (if not provided, all layers are compared)."
        ),
    ] = None,
    jobs: JobsOption = DEFAULT_JOBS,
    distribution_strategy: DistributionStrategyOption = DistributionStrategy.MULTIPROCESSING,
    job_resources: JobResourcesOption = None,
) -> None:
    """Check equality of two WEBKNOSSOS datasets."""

    source_dataset = Dataset.open(source)
    target_dataset = Dataset.open(target)
    source_layer_names = set(source_dataset.layers.keys())
    target_layer_names = set(target_dataset.layers.keys())

    layer_names = list(source_layer_names)

    with get_executor_for_args(
        jobs=jobs,
        distribution_strategy=distribution_strategy,
        job_resources=job_resources,
    ) as executor:
        try:
            if layer_name is not None:
                assert layer_name in source_layer_names, (
                    f"Provided layer {layer_name} does not exist in source dataset."
                )
                assert layer_name in target_layer_names, (
                    f"Provided layer {layer_name} does not exist in target dataset."
                )
                layer_names = [layer_name]

            else:
                assert source_layer_names == target_layer_names, (
                    f"The provided input datasets have different \
    layers: {source_layer_names} != {target_layer_names}"
                )

            for name in layer_names:
                compare_layers(
                    source_dataset.get_layer(name),
                    target_dataset.get_layer(name),
                    executor,
                )

            print(
                f"The datasets {source} and {target} are equal \
    (with regard to the layers: {layer_names})"
            )
        except RuntimeError as err:
            print(f"The datasets are not equal: {err}")
            exit(1)


def compare_layers(
    source_layer: Layer,
    target_layer: Layer,
    executor: Executor,
) -> None:
    """Compares one layer with another layer"""

    layer_name = source_layer.name
    logger.info("Checking layer_name: %s", layer_name)

    if source_layer.bounding_box != target_layer.bounding_box:
        raise RuntimeError(
            f"The bounding boxes of '{layer_name}' layer of source and target \
are not equal: {source_layer.bounding_box} != {target_layer.bounding_box}"
        )

    source_mags = set(source_layer.mags.keys())
    target_mags = set(target_layer.mags.keys())

    if source_mags != target_mags:
        raise RuntimeError(
            f"The mags of '{layer_name}' layer of source and target are not equal: \
{source_mags} != {target_mags}"
        )

    for mag in source_mags:
        source_mag = source_layer.mags[mag]
        target_mag = target_layer.mags[mag]

        logger.info("Start verification of %s in mag %s", layer_name, mag)
        if not source_mag.content_is_equal(
            target_mag,
            executor=executor,
            progress_desc=f"Comparing {layer_name}/{mag}",
        ):
            raise RuntimeError(f"The contents of {source_mag} and {target_mag} differ.")
