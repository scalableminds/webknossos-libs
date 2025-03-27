"""This module takes care of upsampling WEBKNOSSOS datasets."""

from argparse import Namespace
from multiprocessing import cpu_count
from typing import Annotated, Any

import typer

from ..dataset import Dataset, SamplingModes
from ..geometry import Mag
from ..utils import get_executor_for_args
from ._utils import DistributionStrategy, SamplingMode, parse_mag, parse_path


def main(
    *,
    source: Annotated[
        Any,
        typer.Argument(
            help="Path to your WEBKNOSSOS dataset.",
            show_default=False,
            parser=parse_path,
        ),
    ],
    sampling_mode: Annotated[
        SamplingMode, typer.Option(help="The sampling mode to use.")
    ] = SamplingMode.ANISOTROPIC,
    from_mag: Annotated[
        Mag,
        typer.Option(
            help="Mag to start upsampling from. \
Should be number or minus separated string (e.g. 2 or 2-2-2).",
            parser=parse_mag,
        ),
    ],
    layer_name: Annotated[
        str | None,
        typer.Option(
            help="Name of the layer that should be downsampled.", show_default=False
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
        str | None,
        typer.Option(
            help='Necessary when using slurm as distribution strategy. Should be a JSON string \
(e.g., --job-resources=\'{"mem": "10M"}\')\'',
            rich_help_panel="Executor options",
        ),
    ] = None,
) -> None:
    """Upsample your WEBKNOSSOS dataset."""

    executor_args = Namespace(
        jobs=jobs,
        distribution_strategy=distribution_strategy.value,
        job_resources=job_resources,
    )
    dataset = Dataset.open(source)
    mode = SamplingModes.parse(sampling_mode.value)

    if layer_name is None:
        upsample_all_layers(dataset, mode, from_mag, executor_args)
    else:
        with get_executor_for_args(args=executor_args) as executor:
            layer = dataset.get_layer(layer_name)
            layer.upsample(from_mag=from_mag, sampling_mode=mode, executor=executor)


def upsample_all_layers(
    dataset: Dataset, mode: SamplingModes, from_mag: Mag, executor_args: Namespace
) -> None:
    """Iterates over all layers and upsamples them."""

    for layer in dataset.layers.values():
        with get_executor_for_args(args=executor_args) as executor:
            layer.upsample(
                from_mag=from_mag,
                sampling_mode=mode,
                executor=executor,
            )
